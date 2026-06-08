# --- FULL-STATE (4 tubes + 4 columns, 2 comps) EKF ADAPTER -------------------
# Adds: build_A_sequence_full(...) and build_selector_S_full(...)
from typing import List, Tuple, Dict
import numpy as np

import logging
log = logging.getLogger("EKF")

try:
    from SMB.LinColumn import LinColumn
    from SMB.Tube import Tube
except Exception:  # flat layout fallback
    from LinColumn import LinColumn  # type: ignore
    from Tube import Tube            # type: ignore


class CNAdapter:
    """
    Mixin with full-ring builders:
      - build_A_sequence_full(twin, t_start, t_end)
      - build_selector_S_full(twin)
    Can be inherited by your existing CNAdapter or monkey-mixed into it.
    """

    # -------------- PUBLIC API ----------------

    def build_A_sequence_full(self, twin, t_start: float, t_end: float) -> List[np.ndarray]:
        """
        Build a list of A operators for every dt in [t_start, t_end).
        Inserts a Π (permutation) between segments if a switch happens inside.
        """
        import numpy as np

        dt = float(getattr(twin.smb, "dt", twin.params.get("dt", 0.025)))
        if not (t_end > t_start and dt > 0):
            return []

        # Number of CN substeps to cover the window
        Δt = float(t_end - t_start)
        m_total = int(np.floor(Δt / dt + 1e-9))
        if m_total <= 0:
            return []

        # How far into the *current* period are we at t_end?
        # (twin.smb is already at t_end; use its countdown to split)
        # steps remaining in current period AFTER t_end:
        countdown = float(getattr(twin.smb, "countdown", 0.0))
        if countdown < 0:  # no switching → single segment
            segs = [(m_total, 0)]
        else:
            steps_post = int(np.floor(countdown / dt + 1e-9))  # steps of the *next* window after t_end
            # We need to walk *backwards* from t_end to t_start:
            steps_pre = max(0, m_total - steps_post)
            # Number of full periods prior to the immediately previous edge
            steps_per_period = int(max(1, round(float(getattr(twin.smb, "interval", dt)) / dt)))
            segs: List[Tuple[int, int]] = []
            if steps_pre > 0:
                # Could span many previous periods
                k = 1
                to_assign = steps_pre
                while to_assign > 0:
                    take = min(to_assign, steps_per_period)
                    segs.append((take, -k))  # -1 = previous phase, -2 = two back, ...
                    to_assign -= take
                    k += 1
            if steps_post > 0:
                segs.append((steps_post, 0))

            # Now reverse to go forward in time from t_start→t_end
            segs.reverse()

        # Precompute per-phase layouts (object order & block slices) for all needed offsets
        needed_offsets = sorted(set(off for _, off in segs))
        phase_layouts = {off: self._phase_layout_from_ids(twin, off) for off in needed_offsets}

        # Build A for each distinct phase
        A_phase: Dict[int, np.ndarray] = {}
        for off in needed_offsets:
            A_phase[off] = self._build_A_for_phase_full(twin, phase_layouts[off])

        # Assemble sequence + Π between different layouts
        A_seq: List[np.ndarray] = []
        for i, (steps, off) in enumerate(segs):
            if steps <= 0:
                continue
            A_seg = A_phase[off]
            A_seq.extend([A_seg] * steps)

            # If next segment uses a different phase, insert Π mapping off → off_next
            if i + 1 < len(segs):
                _, off_next = segs[i + 1]
                if off_next != off:
                    P = self._build_permutation_between_layouts(phase_layouts[off], phase_layouts[off_next])
                    A_seq.append(P)

        # --- Sanitize sequence so EKF never sees None/bad shapes ---
        N = phase_layouts[needed_offsets[0]].dim
        A_seq_clean = []
        for idx, A in enumerate(A_seq):
            if A is None:
                log.error("[cn] A_seq[%d] is None; inserting identity.", idx)
                A_seq_clean.append(np.eye(N))
                continue
            A = np.asarray(A)
            if A.ndim != 2 or A.shape != (N, N):
                log.error("[cn] A_seq[%d] has bad shape %s; inserting identity.", idx, getattr(A, "shape", None))
                A_seq_clean.append(np.eye(N))
                continue
            if not np.isfinite(A).all():
                log.warning("[cn] A_seq[%d] contains non-finite entries; inserting identity.", idx)
                A_seq_clean.append(np.eye(N))
                continue
            A_seq_clean.append(A)

        # Trim in case Π made us overshoot by one
        return A_seq_clean[:m_total]

    def build_selector_S_full(self, twin) -> np.ndarray:
        """4×N selektor posledních buněk posledních KOLON v Z1 a Z3 (Ex/Ra)."""
        import numpy as np
        L = twin.build_full_state_layout()
        N = L.dim
        i_ex_m, i_ex_g, i_ra_m, i_ra_g = twin.outlet_indices_full(L)
        S = np.zeros((4, N), dtype=float)
        S[0, i_ex_m] = 1.0
        S[1, i_ex_g] = 1.0
        S[2, i_ra_m] = 1.0
        S[3, i_ra_g] = 1.0
        return S

    # -------------- INTERNALS ----------------

    # A compact representation of a phase layout that doesn't depend on object copies
    class _PhaseEntry:
        __slots__ = ("zone", "kind", "obj_id", "slc0", "slc1")
        def __init__(self, zone: int, kind: str, obj_id: int, slc0: slice, slc1: slice) -> None:
            self.zone = zone; self.kind = kind; self.obj_id = obj_id; self.slc0 = slc0; self.slc1 = slc1

    class _PhaseLayout:
        __slots__ = ("dim", "entries", "y_index_by_obj", "zone_index_lists")
        def __init__(self, dim: int, entries: List["CNAdapterFullMixin._PhaseEntry"],
                     y_index_by_obj: Dict[Tuple[int, int], int],
                     zone_index_lists: Dict[int, List[int]]) -> None:
            self.dim = dim
            self.entries = entries              # in flattened order
            self.y_index_by_obj = y_index_by_obj  # (obj_global_idx, comp) -> y row
            self.zone_index_lists = zone_index_lists  # zone -> list of obj_global_idx (in zone order)

    def _phase_layout_from_ids(self, twin, phase_offset: int) -> "_PhaseLayout":
        """
        Build a layout for a phase given only the *ids* of current twin objects, rotated virtually.
        This avoids deep-copy identity issues and lets us place blocks in a consistent vector space.
        """
        import numpy as np

        # Collect current zone lists as ids (keeps Tube,Col ordering)
        zones_ids: Dict[int, List[int]] = {z: [id(o) for o in twin.smb.zones[z]] for z in (1, 2, 3, 4)}

        # Rotate ids *backwards* (-1 = one switch earlier) by applying the inverse of SMBStation.rotate()
        def rotate_ids_backward_once(zids: Dict[int, List[int]]) -> None:
            # Inverse of: Z1→Z4 append; Z2→Z1; Z3→Z2; Z4→Z3
            z1, z2, z3, z4 = zids[1], zids[2], zids[3], zids[4]

            # bring last pair from Z4 to *front* of Z1
            k = min(2, len(z4))
            take = z4[-k:]; del z4[-k:]
            if take:
                z1[0:0] = take

            # shift Z3→Z4
            k = min(2, len(z3))
            take = z3[-k:]; del z3[-k:]
            if take:
                z4.extend(take)

            # shift Z2→Z3
            k = min(2, len(z2))
            take = z2[-k:]; del z2[-k:]
            if take:
                z3.extend(take)

        steps_back = (-int(phase_offset)) % 4
        for _ in range(steps_back):
            rotate_ids_backward_once(zones_ids)

        # Map id -> live object (for sizes/comp vectors)
        id2obj = {id(o): o for z in (1, 2, 3, 4) for o in twin.smb.zones[z]}

        # Build flattened entries with slices
        off = 0
        entries: List[CNAdapter._PhaseEntry] = []
        y_index_by_obj: Dict[Tuple[int, int], int] = {}
        zone_index_lists: Dict[int, List[int]] = {1: [], 2: [], 3: [], 4: []}
        obj_global_idx = 0
        for z in (1, 2, 3, 4):
            for obj_id in zones_ids[z]:
                obj = id2obj[obj_id]
                kind = "tube" if isinstance(obj, Tube) else ("col" if isinstance(obj, LinColumn) else type(obj).__name__)
                c0 = np.asarray(obj.components[0].c, float).reshape(-1)
                c1 = np.asarray(obj.components[1].c, float).reshape(-1)
                sl0 = slice(off, off + c0.size); off += c0.size
                sl1 = slice(off, off + c1.size); off += c1.size
                entries.append(self._PhaseEntry(z, kind, obj_id, sl0, sl1))
                # two rows in y (outlets) per object: (j,0) and (j,1)
                y_index_by_obj[(obj_global_idx, 0)] = obj_global_idx * 2 + 0
                y_index_by_obj[(obj_global_idx, 1)] = obj_global_idx * 2 + 1
                zone_index_lists[z].append(obj_global_idx)
                obj_global_idx += 1

        return self._PhaseLayout(off, entries, y_index_by_obj, zone_index_lists)

    def _build_A_for_phase_full(self, twin, L: "_PhaseLayout") -> np.ndarray:
        """
        One-step global transition A for a given phase layout L, matching SMBStation timing.
        Physics: x_{k+1} = M x_k + U R E x_k  ⇒  A = M + U R E   (delayed mixing).
        """
        import numpy as np
        from typing import Tuple, Dict, List
        from scipy.linalg import lu_solve

        N = L.dim
        m = len(L.entries) * 2  # 2 comps per object (outlet rows)

        # --- Global containers ---
        M = np.zeros((N, N), dtype=float)
        U = np.zeros((N, m), dtype=float)

        # --- object cache ---
        id2obj = {id(o): o for _, _, o in twin._iter_zone_objects()}

        # ---------- Per-object local blocks ----------
        def tube_blocks(n: int, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
            """
            Bounded sub-stepping (α_sub ≤ 1), then collapse:
              T = T_sub^nsub,
              u = (I - T) (I - T_sub)^{-1} e0   (closed form geometric sum).
            """
            I = np.eye(n, dtype=float)
            if n <= 0:
                return I, np.zeros(n, dtype=float)

            nsub = int(np.ceil(max(1.0, float(alpha))))
            a = float(alpha) / nsub  # ≤ 1

            # T_sub = (1-a) I + a ShiftUp
            T_sub = (1.0 - a) * I
            if n > 1 and a > 0.0:
                T_sub[1:, :-1] += a * np.eye(n - 1)

            # power-by-doubling for T = T_sub^nsub (n is small; this is cheap and stable)
            T = np.eye(n, dtype=float)
            B = T_sub.copy()
            k = nsub
            while k:
                if k & 1:
                    T = B @ T
                B = B @ B
                k >>= 1

            # inlet geometric sum (stable linear solve)
            e0 = np.zeros(n, dtype=float); e0[0] = 1.0
            # handle near-singular (I - T_sub) when a ~ 0: fall back to iterative sum
            if a > 1e-12:
                u = (np.eye(n) - T) @ np.linalg.solve(np.eye(n) - T_sub, e0)
            else:
                # tiny flow: approximate by nsub * e0
                u = nsub * e0

            # light renormalization: keep column sums at 1 (except last column, which is the physical outlet)
            colsum = T.sum(axis=0)
            good = colsum[:-1] != 0.0
            T[:, :-1][:, good] /= colsum[:-1][good]

            return T, u

        def column_blocks(col_obj, comp_idx: int, n: int) -> Tuple[np.ndarray, np.ndarray]:
            comp = col_obj.components[comp_idx]
            try:
                Tloc = lu_solve(comp._lu, comp._B)  # A^{-1} B
            except Exception:
                log.exception("[cn] lu_solve(A,B) failed for column id=%s comp=%d; using I.", id(col_obj), comp_idx)
                Tloc = np.eye(n)
            try:
                e0 = np.zeros(n, dtype=float); e0[0] = float(getattr(comp, "_inlet_coef", 0.0))
                uloc = lu_solve(comp._lu, e0)       # A^{-1} e0 * inlet_coef
            except Exception:
                log.exception("[cn] lu_solve(A,e0) failed for column id=%s comp=%d; using e0.", id(col_obj), comp_idx)
                uloc = np.zeros(n, dtype=float); uloc[0] = 1.0

            if Tloc.shape != (n, n) or not np.isfinite(Tloc).all():
                log.error("[cn] malformed column Tloc; using I.")
                Tloc = np.eye(n)
            if uloc.shape != (n,) or not np.isfinite(uloc).all():
                log.error("[cn] malformed column uloc; using e0.")
                uloc = np.zeros(n, dtype=float); uloc[0] = 1.0
            return Tloc, uloc

        # ---------- Place per-object blocks into M and U ----------
        for j, e in enumerate(L.entries):
            obj = id2obj[e.obj_id]
            for comp_idx, slc in ((0, e.slc0), (1, e.slc1)):
                n = int(slc.stop - slc.start)
                if n <= 0:
                    continue
                if isinstance(obj, Tube):
                    alpha = float(getattr(obj, "_alpha", 0.0))
                    Tloc, uloc = tube_blocks(n, alpha)
                elif isinstance(obj, LinColumn):
                    Tloc, uloc = column_blocks(obj, comp_idx, n)
                else:
                    raise TypeError(f"Unexpected object type {type(obj).__name__}")
                M[slc, slc] = Tloc
                U[slc, L.y_index_by_obj[(j, comp_idx)]] = uloc

        # ---------- E: last node of each object/comp ----------
        E = np.zeros((m, N), dtype=float)
        for j, e in enumerate(L.entries):
            E[j * 2 + 0, e.slc0.stop - 1] = 1.0
            E[j * 2 + 1, e.slc1.stop - 1] = 1.0

        # ---------- R and d (feed/recycle split from actual object flow) ----------
        # zone totals still read for alpha source, but denominators come from the *zone's first column/tube*
        Q1 = float(twin.smb.flowRates[1])
        Q2 = float(twin.smb.flowRates[2])
        Q3 = float(twin.smb.flowRates[3])
        # Q4 not used

        cfeed = [float(twin.smb.components[0].feedConc),
                 float(twin.smb.components[1].feedConc)]

        def sdiv(num, den, eps=1e-12):
            d = den if abs(den) > eps else (eps if den >= 0 else -eps)
            return num / d

        # helper: nearest upstream object (global j) before zone z
        def _last_from_prev_zone(zone_lists: Dict[int, List[int]], z: int):
            for back in (1, 2, 3, 4):
                zp = ((z - back - 1) % 4) + 1
                lst = zone_lists.get(zp, [])
                if lst:
                    return lst[-1]
            return None

        # helper: denominator flow for the FIRST object in zone z: prefer first column flow, else tube flow
        def _zone_denominator_flow(z: int, zone_lists: Dict[int, List[int]]) -> float:
            for j in zone_lists.get(z, []):
                o = id2obj[L.entries[j].obj_id]
                if isinstance(o, LinColumn):
                    return float(getattr(o, "flowRate", 0.0))
            for j in zone_lists.get(z, []):
                o = id2obj[L.entries[j].obj_id]
                if isinstance(o, Tube):
                    return float(getattr(o, "flowRate", 0.0))
            return 0.0

        R = np.zeros((m, m), dtype=float)
        d = np.zeros((m,), dtype=float)  # affine part not used in A

        zone_lists = L.zone_index_lists
        for z in (1, 2, 3, 4):
            objs = zone_lists[z]
            for pos, j in enumerate(objs):
                for comp_idx in (0, 1):
                    row = j * 2 + comp_idx
                    if pos > 0:
                        # interior: inlet from previous object in the same zone
                        prev_j = objs[pos - 1]
                        R[row, prev_j * 2 + comp_idx] = 1.0
                    else:
                        # first in zone: recycle (+ feed/eluent where applicable)
                        prev_j = _last_from_prev_zone(zone_lists, z)
                        if prev_j is None:
                            log.error("[cn] No upstream object before zone %d; self-wire to keep graph connected (row=%d).", z, row)
                            R[row, row] = 1.0
                            continue

                        den_flow = _zone_denominator_flow(z, zone_lists)  # <-- key: local denominator
                        col = prev_j * 2 + comp_idx

                        if z == 1:
                            # eluent + recycle: ((Q2/den) * recycle) + ((den-Q2)/den * eluent(=0))
                            alpha = sdiv(Q2, den_flow) if den_flow > 0.0 else 0.0
                            R[row, col] = alpha
                        elif z == 3:
                            # feed + recycle: ((Q2/den) * recycle) + ((den-Q2)/den * feed)
                            alpha = sdiv(Q2, den_flow) if den_flow > 0.0 else 0.0
                            beta  = sdiv(den_flow - Q2, den_flow) if den_flow > 0.0 else 0.0
                            R[row, col] = alpha
                            d[row] = beta * cfeed[comp_idx]
                        else:
                            R[row, col] = 1.0

        # ---------- Global operator ----------
        A = M + U @ R @ E

        # Diagnostics & sanity
        if not np.isfinite(A).all() or A.shape != (N, N):
            log.error("[cn] bad A; returning identity (N=%d).", N)
            return np.eye(N)
        return A

    def _build_permutation_between_layouts(self, L_prev: "_PhaseLayout", L_next: "_PhaseLayout") -> np.ndarray:
        import numpy as np
        if L_prev.dim != L_next.dim:
            raise RuntimeError("Dimension change between layouts; cannot permute safely.")
        N = L_prev.dim
        P = np.zeros((N, N), dtype=float)
        id2dst = {e.obj_id: (e.slc0, e.slc1) for e in L_next.entries}
        for e_prev in L_prev.entries:
            dst0, dst1 = id2dst[e_prev.obj_id]
            P[dst0, e_prev.slc0] = np.eye(dst0.stop - dst0.start)
            P[dst1, e_prev.slc1] = np.eye(dst1.stop - dst1.start)
        return P
