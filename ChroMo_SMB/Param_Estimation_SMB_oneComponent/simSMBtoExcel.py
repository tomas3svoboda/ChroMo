from simSMB_oneComp import simSMB
import pandas as pd

# Define the output file name
output_file = "SMB_Simulation_Man.xlsx"

df_extract, df_raffinate = simSMB(end_time=10000,
                                  name_compA="Man",
                                  henry_constantA=0.51055,
                                  deltaA=22.08344,
                                  name_compB="Gal",
                                  henry_constantB=3.62,
                                  deltaB=28.02,
                                  flow_rates=[180, 93, 114, 45],
                                  switch_interval=780,
                                  optimize_component='A')

# Create a new Excel file and write DataFrames
with pd.ExcelWriter(output_file, mode='w', engine='openpyxl') as writer:
    df_extract.to_excel(writer, sheet_name="Extract", index=False)
    df_raffinate.to_excel(writer, sheet_name="Raffinate", index=False)

print(f"New Excel file '{output_file}' has been created with Extract and Raffinate data.")

