
from opcua import ua, Server
import threading, time
from datetime import datetime

class SMBOPCUAServer(threading.Thread):
    """OPC UA server exposing SMB simulator data.

    Semantics:
      - Existing nodes (FeedFlow, EluentFlow, RecycleFlow, ExtractFlow, SwitchInterval)
        are treated as **Next** (buffered setpoints) to remain backward compatible.
      - New nodes prefixed with Active* expose the setpoints currently driving the plant.
      - The simulator (GUI) copies Next -> Active exactly at switch time.
      - Mode arbitration:
          * manual: server pushes Active* to nodes each cycle; Next* are also written from Active*
                    (so UA clients see coherent values; MPC should not write in manual).
          * automatic: server reads Next* from nodes into shared_state['next_*'] for the plant
                       to apply on switch; Active* are written from shared_state['active_*'].
    """

    def __init__(self, shared_state, endpoint="opc.tcp://127.0.0.1:4840"):
        super().__init__(daemon=True)
        self.shared_state = shared_state
        self.endpoint = endpoint
        self.server = None
        self._running = True
        self._nodes = {}

    def run(self):
        self.server = Server()
        self.server.set_endpoint(self.endpoint)
        self.server.set_security_policy([ua.SecurityPolicyType.NoSecurity])
        idx = self.server.register_namespace("SMB_Simulation")

        obj = self.server.nodes.objects.add_object(idx, "SMB_Simulation")

        # Control/Mode
        mode_node   = obj.add_variable(idx, "Mode", str(self.shared_state.get('mode','manual'))); mode_node.set_writable()
        # Backward-compatible NEXT (writable)
        feed_node   = obj.add_variable(idx, "FeedFlow",   float(self.shared_state.get('feed',0.0)));   feed_node.set_writable()
        eluent_node = obj.add_variable(idx, "EluentFlow", float(self.shared_state.get('eluent',0.0))); eluent_node.set_writable()
        extract_node= obj.add_variable(idx, "ExtractFlow",float(self.shared_state.get('extract',0.0)));extract_node.set_writable()
        recycle_node= obj.add_variable(idx, "RecycleFlow",float(self.shared_state.get('recycle',0.0)));recycle_node.set_writable()
        switch_node = obj.add_variable(idx, "SwitchInterval", float(self.shared_state.get('switch_interval',0.0))); switch_node.set_writable()

        # ACTIVE (read-only for clients)
        act_feed    = obj.add_variable(idx, "ActiveFeedFlow",   float(self.shared_state.get('active_feed',0.0)))
        act_eluent  = obj.add_variable(idx, "ActiveEluentFlow", float(self.shared_state.get('active_eluent',0.0)))
        act_extract = obj.add_variable(idx, "ActiveExtractFlow",float(self.shared_state.get('active_extract',0.0)))
        act_recycle = obj.add_variable(idx, "ActiveRecycleFlow",float(self.shared_state.get('active_recycle',0.0)))
        act_switch  = obj.add_variable(idx, "ActiveSwitchInterval", float(self.shared_state.get('active_switch_interval',0.0)))
        speed_node = obj.add_variable(idx, "SpeedFactor", float(self.shared_state.get('speed_factor', 1.0)))

        # Revisions & diagnostics
        next_rev    = obj.add_variable(idx, "NextRevision",  int(self.shared_state.get('next_revision',0))); next_rev.set_writable()
        last_rev    = obj.add_variable(idx, "LastAppliedRevision", int(self.shared_state.get('last_applied_revision',-1)))
        last_ts     = obj.add_variable(idx, "LastApplyTimestamp",  str(self.shared_state.get('last_apply_timestamp','')))

        # Process values (PVs)
        extract_man_node   = obj.add_variable(idx, "ExtractConcentration_Man",  float(self.shared_state.get('extract_concentration_man',0.0)))
        extract_gal_node   = obj.add_variable(idx, "ExtractConcentration_Gal",  float(self.shared_state.get('extract_concentration_gal',0.0)))
        raffinate_man_node = obj.add_variable(idx, "RaffinateConcentration_Man",float(self.shared_state.get('raffinate_concentration_man',0.0)))
        raffinate_gal_node = obj.add_variable(idx, "RaffinateConcentration_Gal",float(self.shared_state.get('raffinate_concentration_gal',0.0)))
        extract_outlet_flow_node   = obj.add_variable(idx, "ExtractOutletFlow",   float(self.shared_state.get('extract_outlet_flow',0.0)))
        raffinate_outlet_flow_node = obj.add_variable(idx, "RaffinateOutletFlow", float(self.shared_state.get('raffinate_outlet_flow',0.0)))
        time_node           = obj.add_variable(idx, "SimulationTime",   float(self.shared_state.get('simulation_time',0.0)))
        is_running_node     = obj.add_variable(idx, "IsRunning",        bool(self.shared_state.get('is_running',False)))
        switch_countdown_node = obj.add_variable(idx, "SwitchCountdown", float(self.shared_state.get('switch_countdown',0.0)))
        elapsed_wall_node   = obj.add_variable(idx, "ElapsedWallTime",  float(self.shared_state.get('elapsed_wall_time',0.0)))

        # Keep handles
        self._nodes = {
            'Mode': mode_node,
            # NEXT
            'FeedFlow': feed_node, 'EluentFlow': eluent_node, 'ExtractFlow': extract_node,
            'RecycleFlow': recycle_node, 'SwitchInterval': switch_node,
            # ACTIVE
            'ActiveFeedFlow': act_feed, 'ActiveEluentFlow': act_eluent, 'ActiveExtractFlow': act_extract,
            'ActiveRecycleFlow': act_recycle, 'ActiveSwitchInterval': act_switch,
            # Revisions
            'NextRevision': next_rev, 'LastAppliedRevision': last_rev, 'LastApplyTimestamp': last_ts,
            # PVs
            'ExtractConcentration_Man': extract_man_node, 'ExtractConcentration_Gal': extract_gal_node,
            'RaffinateConcentration_Man': raffinate_man_node, 'RaffinateConcentration_Gal': raffinate_gal_node,
            'ExtractOutletFlow': extract_outlet_flow_node, 'RaffinateOutletFlow': raffinate_outlet_flow_node,
            'SimulationTime': time_node, 'IsRunning': is_running_node,
            'SwitchCountdown': switch_countdown_node, 'ElapsedWallTime': elapsed_wall_node,
        }

        self.server.start()
        print(f"OPC UA server started at {self.endpoint}")

        try:
            while self._running:
                s = self.shared_state

                # Always push PVs and diagnostics
                mode_node.set_value(str(s.get('mode','manual')))
                extract_man_node.set_value(float(s.get('extract_concentration_man',0.0)))
                extract_gal_node.set_value(float(s.get('extract_concentration_gal',0.0)))
                raffinate_man_node.set_value(float(s.get('raffinate_concentration_man',0.0)))
                raffinate_gal_node.set_value(float(s.get('raffinate_concentration_gal',0.0)))
                extract_outlet_flow_node.set_value(float(s.get('extract_outlet_flow',0.0)))
                raffinate_outlet_flow_node.set_value(float(s.get('raffinate_outlet_flow',0.0)))
                time_node.set_value(float(s.get('simulation_time',0.0)))
                is_running_node.set_value(bool(s.get('is_running',False)))
                switch_countdown_node.set_value(float(s.get('switch_countdown',0.0)))
                elapsed_wall_node.set_value(float(s.get('elapsed_wall_time',0.0)))

                # Active mirror (read-only outward; we write from shared_state)
                act_feed.set_value(float(s.get('active_feed', s.get('feed',0.0))))
                act_eluent.set_value(float(s.get('active_eluent', s.get('eluent',0.0))))
                act_extract.set_value(float(s.get('active_extract', s.get('extract',0.0))))
                act_recycle.set_value(float(s.get('active_recycle', s.get('recycle',0.0))))
                act_switch.set_value(float(s.get('active_switch_interval', s.get('switch_interval',0.0))))
                last_rev.set_value(int(s.get('last_applied_revision', -1)))
                last_ts.set_value(str(s.get('last_apply_timestamp','')))
                speed_node.set_value(float(self.shared_state.get('speed_factor', 1.0)))

                if s.get('mode','manual') == 'automatic':
                    # Pull **NEXT** from nodes -> shared_state
                    s['feed']            = float(feed_node.get_value())
                    s['eluent']          = float(eluent_node.get_value())
                    s['extract']         = float(extract_node.get_value())
                    s['recycle']         = float(recycle_node.get_value())
                    s['switch_interval'] = float(switch_node.get_value())
                    s['next_revision']   = int(next_rev.get_value())
                else:
                    # MANUAL: push Active -> both Active* and backward-compat NEXT nodes
                    feed_node.set_value(float(s.get('active_feed', s.get('feed',0.0))))
                    eluent_node.set_value(float(s.get('active_eluent', s.get('eluent',0.0))))
                    extract_node.set_value(float(s.get('active_extract', s.get('extract',0.0))))
                    recycle_node.set_value(float(s.get('active_recycle', s.get('recycle',0.0))))
                    switch_node.set_value(float(s.get('active_switch_interval', s.get('switch_interval',0.0))))
                    next_rev.set_value(int(s.get('next_revision', 0)))

                time.sleep(0.1)
        finally:
            self.server.stop()
            print("OPC UA server stopped.")

    def push_from_shared_state(self):
        """Optional one-shot push used by GUI after Apply (manual mode)."""
        if not self._nodes:
            return
        s = self.shared_state
        self._nodes['FeedFlow'].set_value(float(s.get('feed',0.0)))
        self._nodes['EluentFlow'].set_value(float(s.get('eluent',0.0)))
        self._nodes['RecycleFlow'].set_value(float(s.get('recycle',0.0)))
        self._nodes['ExtractFlow'].set_value(float(s.get('extract',0.0)))
        self._nodes['SwitchInterval'].set_value(float(s.get('switch_interval',0.0)))
        self._nodes['NextRevision'].set_value(int(s.get('next_revision',0)))

    def stop(self):
        self._running = False
