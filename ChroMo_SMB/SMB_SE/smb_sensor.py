# smb_sensor.py
# -----------------------------------------------------------------------------
# SMB outlet "sensor" that reads OPC UA and applies ONLY a detection limit.
# Returns outlet pairs for Extract (zone 1) and Raffinate (zone 3) as (Man, Gal).
# -----------------------------------------------------------------------------

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict

from ChroMo_SMB.SMB_PlantSim.opcua_client import SMB_OPCUAClient


@dataclass
class OutletTagMap:
    # Adjust these if your server uses different display names
    extract_man: str = "ExtractConcentration_Man"
    extract_gal: str = "ExtractConcentration_Gal"
    raffinate_man: str = "RaffinateConcentration_Man"
    raffinate_gal: str = "RaffinateConcentration_Gal"

REQUIRED_MEAS_KEYS = {
    "ExtractConcentration_Man",
    "ExtractConcentration_Gal",
    "RaffinateConcentration_Man",
    "RaffinateConcentration_Gal",
}

class SMBOutletSensor:
    """
    Reads plant outlet concentrations from OPC UA and applies:
      - detection limit: values with |y| < detlim are clipped to 0.

    All values are assumed to be in g/L (match detlim units to your plant!).
    """

    def __init__(
        self,
        opc_client: SMB_OPCUAClient,
        tags: OutletTagMap = OutletTagMap(),
        detlim_gpl: float = 0.05,
    ) -> None:
        self.cli = opc_client
        self.tags = tags
        self.detlim = float(detlim_gpl)
        _validate_measurement_contract(self.cli)

    # ---- public API ---------------------------------------------------------

    def read_zone(self, zone: int) -> Optional[tuple]:
        """
        Return (Man, Gal) for given outlet zone:
          zone=1 -> Extract
          zone=3 -> Raffinate
        Returns None on failure.
        """
        y_ex_man, y_ex_gal, y_ra_man, y_ra_gal = self._read_and_preprocess()
        if y_ex_man is None:
            return None
        if zone == 1:
            return (y_ex_man, y_ex_gal)
        if zone == 3:
            return (y_ra_man, y_ra_gal)
        return None  # other zones not measured

    # ---- internals ----------------------------------------------------------

    def _read_snapshot(self) -> Dict[str, float]:
        return self.cli.read_snapshot()

    def _read_and_preprocess(self):
        snap = self._read_snapshot()
        # Explicit lookups so missing tags raise a KeyError caught by caller
        try:
            raw = [
                float(snap[self.tags.extract_man]),
                float(snap[self.tags.extract_gal]),
                float(snap[self.tags.raffinate_man]),
                float(snap[self.tags.raffinate_gal]),
            ]
        except KeyError as e:
            # propagate as None tuple so caller can decide (keeps current API)
            return (None, None, None, None)

        # Detection limit (deadband), units = g/L
        raw = [v if abs(v) >= self.detlim else 0.0 for v in raw]
        return tuple(raw)

def _validate_measurement_contract(cli: SMB_OPCUAClient, req=REQUIRED_MEAS_KEYS) -> None:
    snap = cli.read_snapshot()
    missing = [k for k in req if k not in snap]
    if missing:
        raise RuntimeError(
            "OPC UA contract missing required measurement keys: "
            + ", ".join(sorted(missing))
        )
