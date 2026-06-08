# opcua_client.py
# ---------------------------------------------------------------------
# OPC UA client for the SMB station (supports new F/Q1/Q2/Q4 scheme).
# - Prefers new node names; transparently falls back to legacy names.
# - Returns plain Python types; read_snapshot() also mirrors legacy keys.
# ---------------------------------------------------------------------

from __future__ import annotations
import logging
from typing import Any, Dict, Optional, Tuple

try:
    from opcua import Client, ua
except Exception as exc:  # pragma: no cover
    raise RuntimeError("python-opcua is required. Install with: pip install opcua") from exc

logging.getLogger("opcua").setLevel(logging.WARNING)
log = logging.getLogger("OPCUA")

class SMB_OPCUAClient:
    """
    Objects
      └── SMB_Simulation (preferred)  or  SMB (legacy)
    """

    # Preferred -> candidate browse names (first found wins)
    WRITABLE_CANDIDATES = {
        "Mode": ("Mode",),
        "Feed": ("Feed",),
        "Q1":   ("Q1",),
        "Q2":   ("Q2",),
        "Q4":   ("Q4",),
        "SwitchInterval": ("SwitchInterval",),
    }

    READONLY_CANDIDATES = {
        # ACTIVE mirrors
        "ActiveFeed": ("ActiveFeed",),
        "ActiveQ1": ("ActiveQ1",),
        "ActiveQ2": ("ActiveQ2",),
        "ActiveQ4": ("ActiveQ4",),
        "ActiveSwitchInterval": ("ActiveSwitchInterval",),

        # Telemetry
        "IsRunning": ("IsRunning",),
        "SimulationTime": ("SimulationTime",),
        "ElapsedWallTime": ("ElapsedWallTime",),
        "SwitchCountdown": ("SwitchCountdown",),
        "SpeedFactor": ("SpeedFactor",),  # optional default 1.0
        "SwitchIndex": ("SwitchIndex",),
        "LastSwitchSimTime": ("LastSwitchSimTime",),

        # Measurements
        "ExtractConcentration_Man": ("ExtractConcentration_Man",),
        "ExtractConcentration_Gal": ("ExtractConcentration_Gal",),
        "RaffinateConcentration_Man": ("RaffinateConcentration_Man",),
        "RaffinateConcentration_Gal": ("RaffinateConcentration_Gal",),
    }

    OPTIONAL_KEYS = {"SpeedFactor"}

    def __init__(
        self,
        endpoint: str = "opc.tcp://127.0.0.1:4840",
        obj_browse_name: str = "SMB_Simulation",
        timeout_s: float = 4.0,
    ) -> None:
        self.endpoint = endpoint
        self.obj_browse_name = obj_browse_name
        self.timeout_s = timeout_s

        self.client: Optional[Client] = None
        self._root = None
        self._obj = None
        self._ns_idx: Optional[int] = None
        # caches: preferred key -> (node, actual_browse_name)
        self._w_nodes: Dict[str, Tuple[Any, str]] = {}
        self._r_nodes: Dict[str, Tuple[Any, str]] = {}

    # ------------------------- lifecycle ------------------------- #
    def connect(self) -> None:
        if self.client is not None:
            return
        log.info("[opc] Connecting to %s", self.endpoint)
        c = Client(self.endpoint, timeout=self.timeout_s)
        try:
            c.connect()
            self.client = c
            self._root = c.get_root_node()

            objects = self._root.get_child(["0:Objects"])
            smb_node, smb_ns = self._find_child_by_browse_name(objects, self.obj_browse_name)
            if smb_node is None and self.obj_browse_name != "SMB":
                smb_node, smb_ns = self._find_child_by_browse_name(objects, "SMB")
            if smb_node is None and self.obj_browse_name != "SMB_Simulation":
                smb_node, smb_ns = self._find_child_by_browse_name(objects, "SMB_Simulation")
            if smb_node is None:
                kids = []
                try:
                    for ch in objects.get_children():
                        bn = ch.get_browse_name()
                        kids.append(f"{getattr(bn,'NamespaceIndex',None)}:{getattr(bn,'Name',None)}")
                except Exception:
                    kids = ["<unavailable>"]
                raise RuntimeError(f"Could not find plant object. Children={kids}")

            self._obj = smb_node
            self._ns_idx = smb_ns

            # Resolve writable nodes
            self._w_nodes.clear()
            for pref, candidates in self.WRITABLE_CANDIDATES.items():
                node, used = self._resolve_first(candidates)
                if node is None:
                    log.warning("[opc] writable var '%s' not found (candidates=%s)", pref, candidates)
                else:
                    self._w_nodes[pref] = (node, used)

            # Resolve readonly nodes
            self._r_nodes.clear()
            for pref, candidates in self.READONLY_CANDIDATES.items():
                node, used = self._resolve_first(candidates)
                if node is None:
                    if pref in self.OPTIONAL_KEYS:
                        log.warning("[opc] optional ro var '%s' not found", pref)
                    else:
                        log.warning("[opc] ro var '%s' not found (candidates=%s)", pref, candidates)
                else:
                    self._r_nodes[pref] = (node, used)

            log.info("[opc] Connected to %s", self.endpoint)
        except Exception:
            try:
                c.disconnect()
            except Exception:
                pass
            self.client = None
            raise

    def disconnect(self) -> None:
        if not self.client:
            return
        try:
            self.client.disconnect()
        except Exception:
            pass
        finally:
            self.client = None
            self._root = None
            self._obj = None
            self._w_nodes.clear()
            self._r_nodes.clear()

    # -------------------------- helpers -------------------------- #
    def _resolve_first(self, candidates: Tuple[str, ...]) -> Tuple[Optional[Any], Optional[str]]:
        """Return (node, actual_name) for the first candidate that exists under self._obj."""
        for name in candidates:
            try:
                node = self._obj.get_child([f"{self._ns_idx}:{name}"])
                # Force a quick attribute read to ensure it exists
                node.get_node_class()
                return node, name
            except Exception:
                continue
        return None, None

    @staticmethod
    def _to_py(key: str, value: Any) -> Any:
        if key == "IsRunning":
            try:
                return bool(value)
            except Exception:
                return bool(float(value))
        try:
            return float(value)
        except Exception:
            return value

    @staticmethod
    def _find_child_by_browse_name(parent_node, name: str) -> tuple[Optional[Any], Optional[int]]:
        try:
            for child in parent_node.get_children():
                bn = child.get_browse_name()
                if bn and getattr(bn, "Name", None) == name:
                    return child, getattr(bn, "NamespaceIndex", None)
        except Exception:
            pass
        return None, None

    # ---------------------------- I/O ---------------------------- #
    def read_snapshot(self) -> Dict[str, Any]:
        if self.client is None:
            self.connect()
        snap: Dict[str, Any] = {}

        # RO
        for pref, (node, _used) in list(self._r_nodes.items()):
            if node is None:
                if pref in self.OPTIONAL_KEYS and pref == "SpeedFactor":
                    snap[pref] = 1.0
                continue
            snap[pref] = self._to_py(pref, node.get_value())

        # RW (show staged NEXT setpoints too)
        for pref, (node, _used) in list(self._w_nodes.items()):
            if node is None:
                continue
            snap[pref] = self._to_py(pref, node.get_value())

        if "SpeedFactor" not in snap:
            snap["SpeedFactor"] = 1.0
        return snap

    def write(self, values: Dict[str, Any]) -> None:
        if self.client is None:
            self.connect()
        for k, v in values.items():
            if k not in self.WRITABLE_CANDIDATES:
                raise KeyError(f"Variable '{k}' is not writable or not recognized")
            tup = self._w_nodes.get(k)
            if not tup or tup[0] is None:
                raise KeyError(f"OPC variable not found: {k}")
            node = tup[0]
            if isinstance(v, bool):
                node.set_value(ua.DataValue(ua.Variant(bool(v), ua.VariantType.Boolean)))
            elif isinstance(v, (int, float)):
                node.set_value(ua.DataValue(ua.Variant(float(v), ua.VariantType.Double)))
            else:
                node.set_value(ua.DataValue(ua.Variant(str(v), ua.VariantType.String)))

    # Convenience
    def read(self) -> Dict[str, Any]:
        return self.read_snapshot()

    def time_to_switch_wall(self) -> float:
        """Estimate wall-time to next switch: SwitchCountdown / SpeedFactor."""
        if self.client is None:
            self.connect()
        sc_node = self._r_nodes.get("SwitchCountdown", (None, None))[0]
        if sc_node is None:
            raise KeyError("SwitchCountdown node unavailable")
        sc = float(sc_node.get_value())
        sf = float(self._r_nodes.get("SpeedFactor", (None, None))[0].get_value()) if self._r_nodes.get("SpeedFactor", (None, None))[0] else 1.0
        return sc / max(sf, 1e-9)

    def subscribe_active_changes(self, on_change, keys=None, period_ms: int = 50):
        """
        Subscribe to Active* + timing nodes; call `on_change()` on any change.
        Does not import or reference smb_engine; purely callback-based.
        """
        if self.client is None:
            self.connect()

        # choose nodes (default set covers setpoints + timing)
        keys = tuple(keys or (
            "ActiveFeed", "ActiveQ1", "ActiveQ2", "ActiveQ4",
            "ActiveSwitchInterval", "SwitchCountdown", "SimulationTime"
        ))

        nodes = []
        for k in keys:
            tup = self._r_nodes.get(k)
            if tup and tup[0] is not None:
                nodes.append(tup[0])

        if not nodes:
            raise RuntimeError("No nodes available for subscription")

        # Create (or reuse) a subscription
        try:
            from opcua import ua  # only to ensure module present
        except Exception as exc:
            raise RuntimeError("python-opcua subscription support unavailable") from exc

        class _Handler:
            def datachange_notification(self_inner, node, val, data):
                try:
                    on_change()  # wake engine, debounce upstream if desired
                except Exception:
                    # never throw from handler
                    pass

            def event_notification(self_inner, event):
                pass

        # Store for later unsubscribe
        self._sub_handler = _Handler()
        self._subscription = self.client.create_subscription(period_ms, self._sub_handler)
        self._sub_handles = []
        for n in nodes:
            try:
                h = self._subscription.subscribe_data_change(n)
                self._sub_handles.append(h)
            except Exception:
                # continue; some nodes may be missing
                pass
        return True  # success

    def unsubscribe_active_changes(self):
        """Best-effort unsubscribe."""
        sub = getattr(self, "_subscription", None)
        if sub:
            try:
                for h in getattr(self, "_sub_handles", []) or []:
                    try:
                        sub.unsubscribe(h)
                    except Exception:
                        pass
                sub.delete()
            except Exception:
                pass
        self._subscription = None
        self._sub_handler = None
        self._sub_handles = None

# ----------------------------------------------------------------------#
# Manual test
# ----------------------------------------------------------------------#
if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s] %(message)s")
    cli = SMB_OPCUAClient(endpoint="opc.tcp://127.0.0.1:4840")
    try:
        cli.connect()
        snap = cli.read_snapshot()
        log.info("[test] snapshot: %s", snap)
        tts = cli.time_to_switch_wall()
        log.info("[test] time_to_switch_wall: %.3fs", tts)
    finally:
        cli.disconnect()
