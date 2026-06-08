"""
Bootstrap runner for the EKF state estimator
-------------------------------------------
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from typing import Tuple

# Local imports
try:
    from opc import OpcClient, OpcClientConfig
    from twin import DigitalTwin, DEFAULT_PARAMS, DEFAULT_COMPONENTS, DEFAULT_FLOWS
    from cn_adapter import CNAdapter
    from sensors import SensorModel, SensorConfig
    from ekf_core import EKFCore, EKFConfig
    from manager import EKFManager, ManagerConfig
except Exception as e:  # pragma: no cover
    print("[bootstrap] Import error:", e)
    print("Ensure the project modules are on PYTHONPATH and you're running from the repo root.")
    raise


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EKF state estimator bootstrap")
    # OPC / runtime
    p.add_argument("--opc-endpoint", default="opc.tcp://127.0.0.1:4840", help="OPC UA server endpoint")
    p.add_argument("--opc-timeout", type=float, default=5.0, help="OPC connect/read timeout [s]")
    p.add_argument("--opc-poll", type=float, default=0.5, help="OPC poll period [s]")

    # Twin init (discretization + initial flows)
    p.add_argument("--dt", type=float, default=float(DEFAULT_PARAMS["dt"]), help="SMBStation model step [s]")
    p.add_argument("--nx", type=int, default=int(DEFAULT_PARAMS["Nx"]), help="Axial nodes per last column (Z1/Z3)")
    p.add_argument("--switch-interval", type=float, default=float(DEFAULT_PARAMS["switch_interval"]),
                   help="Initial switch interval [s] for the SMB twin at startup")
    p.add_argument("--feed",    type=float, default=float(DEFAULT_FLOWS["feed"]),    help="Initial FEED flow")
    p.add_argument("--eluent",  type=float, default=float(DEFAULT_FLOWS["eluent"]),  help="Initial ELUENT flow")
    p.add_argument("--recycle", type=float, default=float(DEFAULT_FLOWS["recycle"]), help="Initial RECYCLE flow")
    p.add_argument("--extract", type=float, default=float(DEFAULT_FLOWS["extract"]), help="Initial EXTRACT flow (not used for zone totals)")

    # EKF timing
    p.add_argument("--ekf-period", type=float, default=20.0, help="EKF update period [s]")
    p.add_argument("--catchup-cap-steps", type=int, default=None, help="Max twin steps per loop to bound latency")

    # Sensor model
    p.add_argument("--tau", nargs=4, type=float, default=[0.2, 0.2, 0.2, 0.2], metavar=("ExMan","ExGal","RaMan","RaGal"),
                   help="Sensor time constants per channel [s]")

    # EKF mode
    p.add_argument("--full-state", action="store_true",
               help="Enable full 4-column EKF (uses twin.get/set_profiles_full and CN full sequence)")


    # EKF noise/initials
    p.add_argument("--init-var-profile", type=float, default=1e-4, help="Initial variance for profile states")
    p.add_argument("--init-var-sensor", type=float, default=1e-6, help="Initial variance for sensor states")
    p.add_argument("--qx-rate", type=float, default=1e-8, help="Process noise rate for profiles [units^2/s]")
    p.add_argument("--qz-rate", type=float, default=1e-8, help="Process noise rate for sensors [units^2/s]")
    p.add_argument("--r", nargs=4, type=float, default=[1e-6, 1e-6, 1e-6, 1e-6], metavar=("ExMan","ExGal","RaMan","RaGal"),
                   help="Measurement noise variances per channel")
    p.add_argument("--tau-default", type=float, default=0.2, help="Fallback tau if sensor model doesn't expose per-channel taus")

    # Logging & run control
    p.add_argument("--log", default="INFO", help="Log level (DEBUG, INFO, WARNING, ERROR)")
    p.add_argument("--duration", type=float, default=None, help="Run for a fixed duration [s] then exit (default: until Ctrl+C)")

    return p.parse_args(argv)


def _setup_logging(level_name: str) -> None:
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="[%(asctime)s][%(levelname)s][EKF] %(message)s",
        datefmt="%H:%M:%S",
    )


def _tupl4(v: list[float]) -> Tuple[float, float, float, float]:
    assert len(v) == 4
    return float(v[0]), float(v[1]), float(v[2]), float(v[3])


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    _setup_logging(args.log)
    logging.getLogger("opcua").setLevel(logging.WARNING)
    logging.getLogger("opcua.ua").setLevel(logging.WARNING)
    logging.getLogger("uacrypto").setLevel(logging.WARNING)

    # Construct core objects
    opc = OpcClient(OpcClientConfig(endpoint=args.opc_endpoint, timeout_s=args.opc_timeout))

    # Build a fully initialized twin (independent of OPC)
    params = dict(DEFAULT_PARAMS)
    params["dt"] = args.dt
    params["Nx"] = args.nx
    params["switch_interval"] = args.switch_interval

    streams = dict(DEFAULT_FLOWS)
    streams["feed"] = args.feed
    streams["eluent"] = args.eluent
    streams["recycle"] = args.recycle
    streams["extract"] = args.extract

    twin = DigitalTwin(params=params, components=DEFAULT_COMPONENTS, flows=streams)

    cn = CNAdapter()
    sens = SensorModel(SensorConfig(tau=_tupl4(args.tau)))
    ekf = EKFCore(
        EKFConfig(
            init_var_profile=args.init_var_profile,
            init_var_sensor=args.init_var_sensor,
            qx_rate=args.qx_rate,
            qz_rate=args.qz_rate,
            r_meas_diag=_tupl4(args.r),
            tau_default=args.tau_default,
            full_state=args.full_state,            # <-- NEW
        )
    )

    mgr = EKFManager(
        opc_client=opc,
        twin=twin,
        ekf_core=ekf,
        cn_adapter=cn,
        sensor_model=sens,
        config=ManagerConfig(
            dt_model=args.dt,
            opc_poll=args.opc_poll,
            ekf_period=args.ekf_period,
            catchup_cap_steps=args.catchup_cap_steps,
            csv_path=str(time.time())+"ekf_log.csv",
            full_state=args.full_state,            # <-- NEW
        ),
    )

    # Lifecycle
    def _graceful_shutdown(signum, frame):  # noqa: ARG001
        logging.getLogger("EKF").info("Signal %s received — shutting down…", signum)
        mgr.stop()

    signal.signal(signal.SIGINT, _graceful_shutdown)
    signal.signal(signal.SIGTERM, _graceful_shutdown)

    try:
        opc.connect()
        mgr.start()
        logging.getLogger("EKF").info(
            "Running with endpoint=%s, dt=%.3fs, nx=%d, ekf_period=%.2fs, switch_interval=%.1fs, "
            "flows={feed=%.2f, eluent=%.2f, recycle=%.2f}, full_state=%s",
            args.opc_endpoint, args.dt, args.nx, args.ekf_period, args.switch_interval,
            args.feed, args.eluent, args.recycle, args.full_state
        )
        if args.duration is None:
            while mgr.is_alive():
                time.sleep(0.5)
        else:
            t_end = time.time() + float(args.duration)
            while time.time() < t_end and mgr.is_alive():
                time.sleep(0.1)
    finally:
        mgr.stop()
        mgr.join(timeout=5.0)
        opc.close()
        logging.getLogger("EKF").info("Exited cleanly.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
