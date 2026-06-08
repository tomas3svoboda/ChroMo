from opcua import ua, Server
import threading, time
from datetime import datetime

class SMBOPCUAServer(threading.Thread):
    """
    OPC UA server exposing SMB simulator data for the new control scheme.
    """

    def __init__(self, shared_state, endpoint="opc.tcp://127.0.0.1:4840"):
        super().__init__(daemon=True)
        self.shared_state = shared_state
        self.endpoint = endpoint
        self.server = None
        self._running = True
        self._nodes = {}
        self._last_sent = {}
        self.fast_period_s = 0.05   # 20 Hz for fast tags (Countdown/SimTime/Speed)
        self.slow_every   = 3       # update “slow” tags every 3rd fast tick

    def run(self):
        from opcua import ua, Server
        self.server = Server()
        self.server.set_endpoint(self.endpoint)
        self.server.set_security_policy([ua.SecurityPolicyType.NoSecurity])
        idx = self.server.register_namespace("SMB_Simulation")

        obj = self.server.nodes.objects.add_object(idx, "SMB_Simulation")

        # -------------------------
        # Writable setpoints (NEXT)
        # -------------------------
        mode_node   = obj.add_variable(idx, "Mode",              str(self.shared_state.get('mode', 'manual'))); mode_node.set_writable()
        feed_node   = obj.add_variable(idx, "Feed",              float(self.shared_state.get('feed', 0.0)));    feed_node.set_writable()
        q1_node     = obj.add_variable(idx, "Q1",                float(self.shared_state.get('q1', 0.0)));      q1_node.set_writable()
        q2_node     = obj.add_variable(idx, "Q2",                float(self.shared_state.get('q2', 0.0)));      q2_node.set_writable()
        q4_node     = obj.add_variable(idx, "Q4",                float(self.shared_state.get('q4', 0.0)));      q4_node.set_writable()
        switch_node = obj.add_variable(idx, "SwitchInterval",    float(self.shared_state.get('switch_interval', 0.0))); switch_node.set_writable()

        # -------------------------
        # Active mirrors (read-only)
        # -------------------------
        act_feed   = obj.add_variable(idx, "ActiveFeed",            float(self.shared_state.get('active_feed', 0.0)))
        act_q1     = obj.add_variable(idx, "ActiveQ1",              float(self.shared_state.get('active_q1',   0.0)))
        act_q2     = obj.add_variable(idx, "ActiveQ2",              float(self.shared_state.get('active_q2',   0.0)))
        act_q4     = obj.add_variable(idx, "ActiveQ4",              float(self.shared_state.get('active_q4',   0.0)))
        act_switch = obj.add_variable(idx, "ActiveSwitchInterval",  float(self.shared_state.get('active_switch_interval', 0.0)))

        # -------------------------
        # Telemetry (read-only)
        # -------------------------
        is_running_node       = obj.add_variable(idx, "IsRunning",        bool(self.shared_state.get('is_running', False)))
        sim_time_node         = obj.add_variable(idx, "SimulationTime",   float(self.shared_state.get('simulation_time', 0.0)))
        elapsed_wall_node     = obj.add_variable(idx, "ElapsedWallTime",  float(self.shared_state.get('elapsed_wall_time', 0.0)))
        switch_countdown_node = obj.add_variable(idx, "SwitchCountdown",  float(self.shared_state.get('switch_countdown', 0.0)))
        speed_node            = obj.add_variable(idx, "SpeedFactor",      float(self.shared_state.get('speed_factor', 1.0)))
        switch_idx_node   = obj.add_variable(idx, "SwitchIndex",        int(self.shared_state.get('switch_index', 0)))
        last_sw_sim_node  = obj.add_variable(idx, "LastSwitchSimTime",  float(self.shared_state.get('last_switch_simtime', 0.0)))

        # -------------------------
        # Measurements (read-only)
        # -------------------------
        ex_man  = obj.add_variable(idx, "ExtractConcentration_Man",   float(self.shared_state.get('extract_concentration_man', 0.0)))
        ex_gal  = obj.add_variable(idx, "ExtractConcentration_Gal",   float(self.shared_state.get('extract_concentration_gal', 0.0)))
        ra_man  = obj.add_variable(idx, "RaffinateConcentration_Man", float(self.shared_state.get('raffinate_concentration_man', 0.0)))
        ra_gal  = obj.add_variable(idx, "RaffinateConcentration_Gal", float(self.shared_state.get('raffinate_concentration_gal', 0.0)))

        # Handles
        self._nodes = {
            # NEXT
            'Mode': mode_node, 'Feed': feed_node, 'Q1': q1_node, 'Q2': q2_node, 'Q4': q4_node, 'SwitchInterval': switch_node,
            # ACTIVE
            'ActiveFeed': act_feed, 'ActiveQ1': act_q1, 'ActiveQ2': act_q2, 'ActiveQ4': act_q4, 'ActiveSwitchInterval': act_switch,
            # Telemetry
            'IsRunning': is_running_node, 'SimulationTime': sim_time_node, 'ElapsedWallTime': elapsed_wall_node,
            'SwitchCountdown': switch_countdown_node, 'SpeedFactor': speed_node, 'SwitchIndex': switch_idx_node,
            'LastSwitchSimTime': last_sw_sim_node,
            # Measurements
            'ExtractConcentration_Man': ex_man, 'ExtractConcentration_Gal': ex_gal,
            'RaffinateConcentration_Man': ra_man, 'RaffinateConcentration_Gal': ra_gal,
        }

        self.server.start()
        print(f"OPC UA server started at {self.endpoint}")

        try:
            tick = 0
            next_deadline = time.perf_counter()
            while self._running:
                t0 = time.perf_counter()

                # 1) Freeze a coherent frame for this scan
                s = dict(self.shared_state)  # <- important for atomic-ish frames

                # 2) Decide cadence
                fast_period = self.fast_period_s if bool(s.get('is_running', False)) else 0.2
                is_slow_tick = (tick % self.slow_every == 0)

                # 3) NEXT setpoints (writable) – read or mirror
                mode = str(s.get('mode', 'manual'))
                if mode == 'automatic':
                    # external clients may write these: read from nodes
                    s['feed']            = float(self._nodes['Feed'].get_value())
                    s['q1']              = float(self._nodes['Q1'].get_value())
                    s['q2']              = float(self._nodes['Q2'].get_value())
                    s['q4']              = float(self._nodes['Q4'].get_value())
                    s['switch_interval'] = float(self._nodes['SwitchInterval'].get_value())
                else:
                    # manual: mirror ACTIVE -> NEXT only if changed
                    self._set_if_changed('Feed_NEXT',   self._nodes['Feed'],          float(s.get('active_feed', s.get('feed', 0.0))),        tol=1e-9)
                    self._set_if_changed('Q1_NEXT',     self._nodes['Q1'],            float(s.get('active_q1',   s.get('q1', 0.0))),          tol=1e-9)
                    self._set_if_changed('Q2_NEXT',     self._nodes['Q2'],            float(s.get('active_q2',   s.get('q2', 0.0))),          tol=1e-9)
                    self._set_if_changed('Q4_NEXT',     self._nodes['Q4'],            float(s.get('active_q4',   s.get('q4', 0.0))),          tol=1e-9)
                    self._set_if_changed('SI_NEXT',     self._nodes['SwitchInterval'],float(s.get('active_switch_interval', s.get('switch_interval', 0.0))), tol=1e-9)

                # 4) Telemetry (fast path first)
                self._set_if_changed('Mode',           self._nodes['Mode'],            mode)
                self._set_if_changed('IsRunning',      self._nodes['IsRunning'],       bool(s.get('is_running', False)))
                self._set_if_changed('SimulationTime', self._nodes['SimulationTime'],  self._float(s,'simulation_time'), tol=1e-9)
                self._set_if_changed('SwitchCountdown',self._nodes['SwitchCountdown'], self._float(s,'switch_countdown'), tol=1e-9)
                self._set_if_changed('SpeedFactor',    self._nodes['SpeedFactor'],     self._float(s,'speed_factor'),    tol=1e-12)
                if is_slow_tick:
                    self._set_if_changed('ElapsedWallTime', self._nodes['ElapsedWallTime'], self._float(s,'elapsed_wall_time'), tol=1e-6)

                # 5) ACTIVE mirrors – publish BEFORE switch identity (coherent frame)
                #    (order matters: Active* first, then index+timestamp)
                self._set_if_changed('ActiveFeed',           self._nodes['ActiveFeed'],           self._float(s,'active_feed'),            tol=1e-9)
                self._set_if_changed('ActiveQ1',             self._nodes['ActiveQ1'],             self._float(s,'active_q1'),              tol=1e-9)
                self._set_if_changed('ActiveQ2',             self._nodes['ActiveQ2'],             self._float(s,'active_q2'),              tol=1e-9)
                self._set_if_changed('ActiveQ4',             self._nodes['ActiveQ4'],             self._float(s,'active_q4'),              tol=1e-9)
                self._set_if_changed('ActiveSwitchInterval', self._nodes['ActiveSwitchInterval'], self._float(s,'active_switch_interval'), tol=1e-9)

                # 6) Measurements (can be decimated)
                if is_slow_tick:
                    self._set_if_changed('ExtractConcentration_Man', self._nodes['ExtractConcentration_Man'], self._float(s,'extract_concentration_man'), tol=1e-9)
                    self._set_if_changed('ExtractConcentration_Gal', self._nodes['ExtractConcentration_Gal'], self._float(s,'extract_concentration_gal'), tol=1e-9)
                    self._set_if_changed('RaffinateConcentration_Man', self._nodes['RaffinateConcentration_Man'], self._float(s,'raffinate_concentration_man'), tol=1e-9)
                    self._set_if_changed('RaffinateConcentration_Gal', self._nodes['RaffinateConcentration_Gal'], self._float(s,'raffinate_concentration_gal'), tol=1e-9)


                # 7) Switch identity LAST (coherent with Active*)
                self._set_if_changed('SwitchIndex',       self._nodes['SwitchIndex'],       self._int(s, 'switch_index'))
                self._set_if_changed('LastSwitchSimTime', self._nodes['LastSwitchSimTime'], self._float(s, 'last_switch_simtime'), tol=1e-9)


                # 8) Timing: precise deadline-based sleep
                tick += 1
                next_deadline += fast_period
                sleep_for = max(0.0, next_deadline - time.perf_counter())
                time.sleep(sleep_for)

        finally:
            self.server.stop()
            print("OPC UA server stopped.")

    def push_from_shared_state(self):
        """Optional one-shot push used by GUI after Apply (manual mode)."""
        if not self._nodes:
            return
        s = self.shared_state
        self._nodes['Feed'].set_value(float(s.get('feed', 0.0)))
        self._nodes['Q1'].set_value(float(s.get('q1', 0.0)))
        self._nodes['Q2'].set_value(float(s.get('q2', 0.0)))
        self._nodes['Q4'].set_value(float(s.get('q4', 0.0)))
        self._nodes['SwitchInterval'].set_value(float(s.get('switch_interval', 0.0)))

    def _set_if_changed(self, key, node, val, tol=0.0):
        """Only push to server if value changed (with optional tolerance)."""
        prev = self._last_sent.get(key, None)
        changed = False
        if isinstance(val, float):
            changed = (prev is None) or (abs(float(prev) - float(val)) > tol)
        else:
            changed = (prev != val)
        if changed:
            node.set_value(val)
            self._last_sent[key] = val

    def _float(self, s, k, default=0.0):
        try:
            return float(s.get(k, default))
        except Exception:
            return float(default)

    def _int(self, s, k, default=0):
        try:
            return int(s.get(k, default))
        except Exception:
            try:
                return int(float(s.get(k, default)))
            except Exception:
                return int(default)

    def pull_next_setpoints_into_shared_state(self):
        """Read writable OPC UA nodes immediately into shared_state."""
        if not self._nodes:
            return
        s = self.shared_state
        s['feed']            = float(self._nodes['Feed'].get_value())
        s['q1']              = float(self._nodes['Q1'].get_value())
        s['q2']              = float(self._nodes['Q2'].get_value())
        s['q4']              = float(self._nodes['Q4'].get_value())
        s['switch_interval'] = float(self._nodes['SwitchInterval'].get_value())

    def stop(self):
        self._running = False
