"""
OPC adapter — thin, read‑only wrapper over SMB_OPCUAClient
-----------------------------------------------------------

Purpose
  • Provide a tiny, dependency‑free interface the EKF manager can use.
  • Read only. No writes to the plant.
  • Normalize data types and guarantee required keys exist.
  • Add structured DEBUG logs for testing.

Usage
  from opc import OpcClient
  opc = OpcClient(endpoint="opc.tcp://127.0.0.1:4840")
  opc.connect()
  snap = opc.read()   # dict keyed by tag names
  opc.close()

Notes
  – This file depends on your existing 'opcua_client.SMB_OPCUAClient'.
  – We intentionally return a plain dict so EKFManager can wrap it into
    its OpcSnapshot (or accept the dict directly).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional
import logging

try:
    # Same package
    from ChroMo_SMB.opcua_client import SMB_OPCUAClient
except Exception:  # pragma: no cover
    # Flat layout fallback for quick testing
    from opcua_client import SMB_OPCUAClient  # type: ignore

LOGGER_NAME = "EKF"
log = logging.getLogger(LOGGER_NAME)


# Canonical tag list the EKF expects
REQUIRED_TAGS = [
    # status & timing
    "IsRunning",
    "SimulationTime",
    "SwitchCountdown",
    "ElapsedWallTime",
    "SpeedFactor",
    # operating params
    "FeedFlow",
    "ExtractFlow",
    "RecycleFlow",
    "SwitchInterval",
    "EluentFlow",
    # concentrations
    "ExtractConcentration_Man",
    "ExtractConcentration_Gal",
    "RaffinateConcentration_Man",
    "RaffinateConcentration_Gal",
    # auxiliary
    "ExtractOutletFlow",
    "RaffinateOutletFlow",
]


@dataclass
class OpcClientConfig:
    endpoint: str = "opc.tcp://127.0.0.1:4840"
    timeout_s: float = 5.0


class OpcClient:
    """Read‑only adapter around SMB_OPCUAClient."""

    def __init__(self, config: Optional[OpcClientConfig] = None) -> None:
        cfg = config or OpcClientConfig()
        self._cfg = cfg
        self._cli = SMB_OPCUAClient(endpoint=cfg.endpoint, timeout_s=cfg.timeout_s)
        self._connected = False

    # --------------- lifecycle ---------------
    def connect(self) -> None:
        if self._connected:
            return
        self._cli.connect()
        self._connected = True
        log.info("[opc] Connected to %s", self._cfg.endpoint)

    def close(self) -> None:
        if not self._connected:
            return
        try:
            self._cli.disconnect()
        finally:
            self._connected = False
            log.info("[opc] Disconnected")

    # Context manager helpers
    def __enter__(self) -> "OpcClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # --------------- I/O ---------------
    def read(self) -> Dict[str, Any]:
        if not self._connected:
            raise RuntimeError("OPC client not connected. Call connect().")
        raw = self._cli.read_snapshot()  # returns a dict
        snap = self._normalize(raw)
        log.debug("[opc] snapshot %s", {k: snap[k] for k in REQUIRED_TAGS})
        return snap

    # --------------- internals ---------------
    @staticmethod
    def _normalize(raw: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure all required keys exist and cast to basic Python types.
        Missing keys raise KeyError to surface server mismatches early.
        """
        out: Dict[str, Any] = {}
        for k in REQUIRED_TAGS:
            if k not in raw:
                raise KeyError(f"OPC snapshot missing key: {k}")
            v = raw[k]
            # bools stay bool, numeric to float
            if isinstance(v, bool):
                out[k] = bool(v)
            else:
                try:
                    out[k] = float(v)
                except Exception:
                    # keep as‑is if not castable (rare/impossible for our tags)
                    out[k] = v
        return out


# ----------------- quick smoke test -----------------
if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
    with OpcClient() as opc:
        s = opc.read()
        print({k: s[k] for k in REQUIRED_TAGS})
