import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.signal
from scipy.interpolate import interp1d
from simSMB_oneComp import simSMB

def objectiveSMB(end_time, experimental_data_path, porosity=0.3756, name_compA="Man", henry_constantA=6.36, deltaA=13.51,
                  name_compB="Gal", henry_constantB=3.62, deltaB=28.02, switch_interval=780,
                  optimize_component='Man', save_plot=False, plotting=False):
    """
    Simulates SMB process and compares model results with experimental data.
    Optionally generates plots and exports model data to an Excel file.
    """
    flow_rates = [180, 93, 114, 45]  # Define flow rates for SMB process

    # Run SMB simulation and obtain model results
    df_extract, df_raffinate = simSMB(end_time,
                                      porosity,
                                      name_compA,
                                      henry_constantA,
                                      deltaA,
                                      name_compB,
                                      henry_constantB,
                                      deltaB,
                                      switch_interval=switch_interval,
                                      optimize_component=optimize_component,
                                      flow_rates=flow_rates,
                                      plotting=plotting)

    # Load experimental data from provided Excel file
    df_exp = pd.read_excel(experimental_data_path)

    # Identify last full switching period
    max_time = df_extract['time'].max()
    last_full_period_start = max_time - (max_time % switch_interval) - switch_interval
    last_full_period_end = last_full_period_start + switch_interval

    # Extract data for the last period
    df_extract_last = df_extract[(df_extract['time'] >= last_full_period_start) & (df_extract['time'] < last_full_period_end)].copy()
    df_raffinate_last = df_raffinate[(df_raffinate['time'] >= last_full_period_start) & (df_raffinate['time'] < last_full_period_end)].copy()

    # Adjust time to start from zero within the last period
    df_extract_last['time'] -= last_full_period_start
    df_raffinate_last['time'] -= last_full_period_start

    # Convert experimental time from minutes to seconds
    df_exp['Time_Ext (min)'] *= 60
    df_exp['Time_Raff (min)'] *= 60

    # Forcing double maxima in raff
    # Detect local maxima
    peaks, _ = scipy.signal.find_peaks(df_raffinate_last["concentration"], prominence=0.1)  # Adjust prominence as needed

    # Count number of peaks
    num_peaks = len(peaks)

    print(f"Number of maxima: {num_peaks}")

    # Extract experimental data based on optimized component
    if optimize_component == name_compA:
        exp_extract = df_exp['Man_Ext (g/L)']
        exp_raffinate = df_exp['Man_Raff (g/L)']
    else:
        exp_extract = df_exp['Gal_Ext (g/L)']
        exp_raffinate = df_exp['Gal_Raff (g/L)']

    # Interpolate model data to match experimental time points
    interp_extract = interp1d(df_extract_last['time'], df_extract_last['concentration'], kind='linear', fill_value='extrapolate')
    interp_raffinate = interp1d(df_raffinate_last['time'], df_raffinate_last['concentration'], kind='linear', fill_value='extrapolate')

    # Compute simulated concentration at experimental time points
    sim_extract = interp_extract(df_exp['Time_Ext (min)'])
    sim_raffinate = interp_raffinate(df_exp['Time_Raff (min)'])

    # Compute Mean Squared Error (MSE) for model vs. experimental data
    mse_extract = np.mean((exp_extract - sim_extract) ** 2) / exp_extract.max() # Normalized by maximal conc in extract
    mse_raffinate = np.mean((exp_raffinate - sim_raffinate) ** 2) / exp_raffinate.max() * 1000 # Normalized by maximal conc in raffinate
    objective_total = mse_extract + mse_raffinate
    if num_peaks < 2:
        objective_total += 1000
    print (f"  Actual MSE: {objective_total}" + f";  Actual MSE Ext: {mse_extract}" + f";  Actual MSE Raff: {mse_raffinate}")

    if save_plot:
        """Generate plots and save model data when save_plot is True"""
        fig, axes = plt.subplots(2, 1, figsize=(8, 6))

        # Plot Extract concentration
        axes[0].plot(df_extract_last['time'], df_extract_last['concentration'], label='Model Extract', linestyle='--')
        axes[0].scatter(df_exp['Time_Ext (min)'], exp_extract, color='red', label='Experimental Extract')
        axes[0].set_title(f'Extract Concentration - {optimize_component}')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Concentration (g/L)')
        axes[0].legend()

        # Plot Raffinate concentration
        axes[1].plot(df_raffinate_last['time'], df_raffinate_last['concentration'], label='Model Raffinate', linestyle='--')
        axes[1].scatter(df_exp['Time_Raff (min)'], exp_raffinate, color='red', label='Experimental Raffinate')
        axes[1].set_title(f'Raffinate Concentration - {optimize_component}')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Concentration (g/L)')
        axes[1].legend()

        # Save plot as PNG
        plt.tight_layout()
        plt.savefig(f'concentration_comparison_{optimize_component}.png', dpi=300)
        plt.close()

        # Save model data to Excel
        output_filename = f'model_data_{optimize_component}.xlsx'
        with pd.ExcelWriter(output_filename) as writer:
            df_extract_last.to_excel(writer, sheet_name='Extract Model', index=False)
            df_raffinate_last.to_excel(writer, sheet_name='Raffinate Model', index=False)
        print(f"Model data saved to {output_filename}")

    return objective_total
