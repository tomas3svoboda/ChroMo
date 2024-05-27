import pandas as pd

# Load the Excel file
file_path = 'sensitivity_analysis_results_20240522_164530_Glu.xlsx'  # Update this with your actual file path
df = pd.read_excel(file_path, sheet_name='Sheet1')

# Extracting the list of experiments
experiments = df.columns[4:]

# Creating a dictionary to hold the pivot tables for each experiment
pivot_tables = {}

# Generating pivot tables for each experiment
for experiment in experiments:
    pivot_tables[experiment] = df.pivot_table(index='saturCoef', columns='langmuirConst', values=experiment)

# Saving all pivot tables to a new Excel file with each experiment as a separate sheet
output_file_path = 'all_experiments_pivot_tables.xlsx'
with pd.ExcelWriter(output_file_path) as writer:
    for experiment, table in pivot_tables.items():
        table.to_excel(writer, sheet_name=experiment)

print(f"Pivot tables saved to {output_file_path}")
