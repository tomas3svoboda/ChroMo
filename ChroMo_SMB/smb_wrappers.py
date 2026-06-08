# smb_wrappers.py
class OutletBiasWrapper:
    def __init__(self, smb_station, bias_vec, man_idx=0, gal_idx=1):
        self._smb = smb_station
        self._b = bias_vec.copy() if bias_vec is not None else None
        self._man = man_idx; self._gal = gal_idx

    def deepCopy(self):
        return OutletBiasWrapper(self._smb.deepCopy(), (self._b.copy() if self._b is not None else None),
                                 self._man, self._gal)

    def __getattr__(self, name):
        return getattr(self._smb, name)

    def step_fast_outlets(self):
        c_ex_man, c_ex_gal, c_ra_man, c_ra_gal = self._smb.step_fast_outlets()
        if self._b is None: return c_ex_man, c_ex_gal, c_ra_man, c_ra_gal
        return (c_ex_man + float(self._b[0]), c_ex_gal + float(self._b[1]),
                c_ra_man + float(self._b[2]), c_ra_gal + float(self._b[3]))
