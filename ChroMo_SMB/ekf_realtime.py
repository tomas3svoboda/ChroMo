# ekf_realtime.py
import threading, time
import numpy as np

class OutletBiasEKF:  # tiny EKF just for outlet bias (upgrade later)
    def __init__(self, q=1e-3, r=5e-3, init_var=1e-2):
        self.b = np.zeros(4)                 # [Ex.Man, Ex.Gal, Ra.Man, Ra.Gal]
        self.P = np.eye(4) * init_var
        self.Q = np.eye(4) * q
        self.R = np.eye(4) * r

    def predict(self):
        self.P = self.P + self.Q             # b_{k+1}^- = b_k + w, so P^- = P+Q

    def update(self, y_meas, y_pred):
        H = np.eye(4)                        # y = y_pred + b + v
        innov = y_meas - (y_pred + self.b)
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.b = self.b + K @ innov
        self.P = (np.eye(4) - K @ H) @ self.P

    def snapshot_bias(self):
        return self.b.copy()

class SMBRealTimeEstimator(threading.Thread):
    """
    Runs side-by-side with the plant:
      - steps the SMB model by dt
      - reads y_meas from OPC
      - EKF update → corrected internal state
    Exposes thread-safe snapshot of the current corrected SMB template + bias.
    """
    def __init__(self, smb_template, opc_client, *, use_bias=True):
        super().__init__(daemon=True)
        self._opc = opc_client
        self._dt = float(smb_template.settings["dt"])
        self._steps_per_period = int(round(float(smb_template.interval)/self._dt))
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._running.set()
        # working copy that we keep synchronized with plant:
        self._smb = smb_template.deepCopy()
        self._use_bias = use_bias
        self._ekf = OutletBiasEKF() if use_bias else None

    def stop(self): self._running.clear()

    def run(self):
        use_fast = hasattr(self._smb, "step_fast_outlets")
        while self._running.is_set():
            # ---- model predict by dt ----
            if use_fast:
                y_pred = np.array(self._smb.step_fast_outlets(), float)  # [ExM,ExG,RaM,RaG]
            else:
                res = self._smb.step(1)  # one dt
                y_pred = np.array([
                    self._last_outlet(res, 1, 0),
                    self._last_outlet(res, 1, 1),
                    self._last_outlet(res, 3, 0),
                    self._last_outlet(res, 3, 1),
                ], float)

            # ---- read measurement and EKF update ----
            try:
                snap = self._opc.read_snapshot()
                y_meas = np.array([
                    float(snap["ExtractConcentration_Man"]),
                    float(snap["ExtractConcentration_Gal"]),
                    float(snap["RaffinateConcentration_Man"]),
                    float(snap["RaffinateConcentration_Gal"]),
                ], float)

                if self._ekf:
                    self._ekf.predict()
                    self._ekf.update(y_meas, y_pred)
            except Exception:
                pass  # if measurement not available, just keep predicting

            # pacing roughly with wall-time (optional)
            time.sleep(self._dt * 0.5)  # adjust for your sim/OPC cadence

    def snapshot_for_optimizer(self):
        """
        Return a deep-copied, frozen seed for the optimizer:
         - the corrected SMB template as of 'now'
         - a copy of the outlet-bias vector (if used)
        """
        with self._lock:
            smb_seed = self._smb.deepCopy()
            bias = self._ekf.snapshot_bias() if self._ekf else None
            return smb_seed, bias

    @staticmethod
    def _last_outlet(res, zone, comp_idx):
        arr = res[zone][-1][comp_idx]
        return float(arr[-1])
