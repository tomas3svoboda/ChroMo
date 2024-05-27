import pandas as pd

file_path = 'optimization_run_details.csv'
optim_run_details_df = pd.read_csv(file_path)
# Add 'Call Number' column as a cumulative count
optim_run_details_df['Call Number'] = range(1, len(optim_run_details_df) + 1)

import matplotlib.pyplot as plt

# Set the font to Times New Roman for all text in the plot
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# Create a smaller figure size
plt.figure(figsize=(8, 4))

# Plot the loss function value against the number of calls
plt.plot(optim_run_details_df['Call Number'], optim_run_details_df['Objective Value'], color='b', linestyle='-')

# Set labels and title
plt.xlabel('Number of Calls')
plt.ylabel('Loss Function Value')
plt.title('Evolution of Loss Function Value Over Number of Calls')

# Set the y-axis to start at 0.2 and x-axis to start at 0
plt.ylim(bottom=0.2)
plt.xlim(left=0)

# Remove the grid
plt.grid(False)

# Remove the top and right spines (frame of the plot)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

# Keep the bottom and left spines (x and y axis lines)
plt.gca().spines['bottom'].set_visible(True)
plt.gca().spines['left'].set_visible(True)

# Remove the legend
# Since no legend is needed, we don't have to add or remove it.

# Show the plot
plt.show()
