import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from simSMB import simSMB


def _select_last_full_period(df, switch_interval):
    max_t = float(df['time'].max())
    n_full = int(max_t // switch_interval)
    if n_full < 1:
        raise ValueError(
            f"Not enough simulated time ({max_t:.1f} s) for one full switch interval ({switch_interval} s). "
            f"Increase end_time."
        )
    start = (n_full - 1) * switch_interval
    end = n_full * switch_interval
    out = df.loc[(df['time'] >= start) & (df['time'] < end)].copy()
    out['time'] -= start
    return out


def objectiveSMB(
    end_time,
    experimental_data_path,
    # names & feeds
    name_compA="Man", feed_concA=7.27,
    name_compB="Gal", feed_concB=3.42,
    # params to FIT
    K_A=4.0, qm_A=50.0,
    K_B=7.0, qm_B=50.0,
    delta_shared=20.0,
    # fixed ops / numerics
    Di=7.0e-4,
    flow_rates=None,
    switch_interval=780,
    dt=1, Nx=100,
    verbose=False,
):
    """
    Run nonlinear SMB, take the *last full switch period*, interpolate to experimental points,
    and compute MSE over {Extract A/B, Raffinate A/B}.
    """
    # 1) simulate once
    df_ext, df_raf = simSMB(
        end_time=end_time,
        name_compA=name_compA, feed_concA=feed_concA,
        name_compB=name_compB, feed_concB=feed_concB,
        K_A=K_A, qm_A=qm_A, K_B=K_B, qm_B=qm_B,
        delta_shared=delta_shared,
        Di=Di,
        flow_rates=flow_rates,
        switch_interval=switch_interval,
        dt=dt, Nx=Nx,
        verbose=False,
    )

    # 2) last full switch period
    ext_last = _select_last_full_period(df_ext, switch_interval)
    raf_last = _select_last_full_period(df_raf, switch_interval)

    # 3) load experimental data; convert min→s for time columns
    df_exp = pd.read_excel(experimental_data_path).copy()

    # expected columns:
    #   'Time_Ext (min)', 'Man_Ext (g/L)', 'Gal_Ext (g/L)'
    #   'Time_Raff (min)', 'Man_Raff (g/L)', 'Gal_Raff (g/L)'
    df_exp['Time_Ext (min)']  = df_exp['Time_Ext (min)']  * 60.0
    df_exp['Time_Raff (min)'] = df_exp['Time_Raff (min)'] * 60.0

    # 4) interpolators over last model period
    f_ext_A = interp1d(ext_last['time'], ext_last['concentration_A'], kind='linear', bounds_error=False, fill_value='extrapolate')
    f_ext_B = interp1d(ext_last['time'], ext_last['concentration_B'], kind='linear', bounds_error=False, fill_value='extrapolate')
    f_raf_A = interp1d(raf_last['time'], raf_last['concentration_A'], kind='linear', bounds_error=False, fill_value='extrapolate')
    f_raf_B = interp1d(raf_last['time'], raf_last['concentration_B'], kind='linear', bounds_error=False, fill_value='extrapolate')

    # 5) sample at experimental timestamps
    sim_ext_A = f_ext_A(df_exp['Time_Ext (min)'].to_numpy())
    sim_ext_B = f_ext_B(df_exp['Time_Ext (min)'].to_numpy())
    sim_raf_A = f_raf_A(df_exp['Time_Raff (min)'].to_numpy())
    sim_raf_B = f_raf_B(df_exp['Time_Raff (min)'].to_numpy())

    # 6) MSE over the four signals
    mse_extract_A = np.mean((df_exp['Man_Ext (g/L)'].to_numpy()  - sim_ext_A) ** 2)
    mse_extract_B = np.mean((df_exp['Gal_Ext (g/L)'].to_numpy()  - sim_ext_B) ** 2)
    mse_raff_A    = np.mean((df_exp['Man_Raff (g/L)'].to_numpy() - sim_raf_A) ** 2)
    mse_raff_B    = np.mean((df_exp['Gal_Raff (g/L)'].to_numpy() - sim_raf_B) ** 2)

    mse_total = (mse_extract_A + mse_extract_B + mse_raff_A + mse_raff_B) / 4.0

    if verbose:
        print(f"MSE: total={mse_total:.6g} | ExtA={mse_extract_A:.6g} ExtB={mse_extract_B:.6g} "
              f"RafA={mse_raff_A:.6g} RafB={mse_raff_B:.6g}")

    return float(mse_total)
