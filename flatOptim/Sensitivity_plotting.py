import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Step 1: Read the Excel File
input_file = 'sensitivity_analysis_results_20240522_164530_Glu.xlsx'
df = pd.read_excel(input_file)

# Step 2: Prepare the Data
X = df['langmuirConst'].values
Y = df['saturCoef'].values
Z = df['total_loss'].values

# Reshape the data to a grid (assuming regular grid of x and y values)
x_unique = np.unique(X)
y_unique = np.unique(Y)
X_grid, Y_grid = np.meshgrid(x_unique, y_unique)
Z_grid = Z.reshape(len(y_unique), len(x_unique))

# Step 3: Plot the Surface
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Create the surface plot
surf = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='viridis')

# Set axis limits
ax.set_xlim([X.min(), X.max()])  # Example: [1.5, 2.5]
ax.set_ylim([Y.min(), Y.max()])  # Example: [18, 22]
ax.set_zlim([Z.min(), 150])  # Example: [0, 100] or any appropriate limit based on your data

# Add labels and title
ax.set_xlabel('Langmuir Coefficient')
ax.set_ylabel('Saturation Constant')
ax.set_zlabel('Loss')
ax.set_title('Sensitivity Analysis Surface Plot')

# Add color bar which maps values to colors
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# Show plot
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Step 1: Read the Excel File
input_file = 'sensitivity_analysis_results_20240522_135331_Suc.xlsx'
df = pd.read_excel(input_file)

# Step 2: Prepare the Data
X = df['langmuirConst'].values
Y = df['saturCoef'].values
Z = df['total_loss'].values

# Reshape the data to a grid (assuming regular grid of x and y values)
x_unique = np.unique(X)
y_unique = np.unique(Y)
X_grid, Y_grid = np.meshgrid(x_unique, y_unique)
Z_grid = Z.reshape(len(y_unique), len(x_unique))

# Step 3: Plot the Heat Map
plt.figure(figsize=(8, 6))
heatmap = plt.imshow(Z_grid, extent=(X.min(), X.max(), Y.min(), Y.max()), origin='lower', aspect='auto', cmap='viridis', vmin=0, vmax=15)

# Add labels and title
plt.xlabel('Langmuir Coefficient')
plt.ylabel('Saturation Constant')
plt.title('Sensitivity Analysis Heat Map')

# Add color bar which maps values to colors
plt.colorbar(heatmap, label='Loss')

# Show plot
plt.show()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Provided actual data
objective_values = [
    7.220670315, 6.436793975, 5.807954215, 5.307095047, 4.907600006, 4.589963074, 4.340437947, 4.148640192, 4.00611752,
    3.905717571, 3.841304636, 3.807606511, 3.800115256, 3.815007172, 3.84905947, 3.899558093, 3.964205549, 4.041039862,
    4.128368856, 4.224720345, 4.328805017, 4.43948695, 4.555762772, 4.676746411, 4.801654519, 4.929793935, 5.060553141,
    5.1933933, 5.327839961, 5.463476356, 5.599937722, 5.736905328, 5.874100715, 6.011281271, 6.148236414, 6.284783566,
    6.420764013, 6.556039007, 6.556039007, 6.690486805, 6.823999588, 6.956479389, 7.087833584, 7.217969341, 7.346787647,
    7.474178143, 7.600016479, 7.724167556, 7.84649393
]
Bo_values = [
    10, 12.0212766, 14.04255319, 16.06382979, 18.08510638, 20.10638298, 22.12765957, 24.14893617, 26.17021277,
    28.19148936, 30.21276596, 32.23404255, 34.25531915, 36.27659574, 38.29787234, 40.31914894, 42.34042553, 44.36170213,
    46.38297872, 48.40425532, 50.42553191, 52.44680851, 54.46808511, 56.4893617, 58.5106383, 60.53191489, 62.55319149,
    64.57446809, 66.59574468, 68.61702128, 70.63829787, 72.65957447, 74.68085106, 76.70212766, 78.72340426, 80.74468085,
    82.76595745, 84.78723404, 84.78723404, 86.80851064, 88.82978723, 90.85106383, 92.87234043, 94.89361702, 96.91489362,
    98.93617021, 100.9574468, 102.9787234, 105
]

# Convert objective function values to likelihood values
profile_likelihood_values = np.exp(-0.5 * np.array(objective_values))

# Find the maximum likelihood value
L_max_actual = np.max(profile_likelihood_values)

# Normalize the likelihood values
normalized_likelihoods = profile_likelihood_values / L_max_actual

# Calculate the likelihood ratio test statistic
likelihood_ratios = -2 * np.log(normalized_likelihoods)

# Define the critical value for a 95% confidence interval (chi-squared distribution with 1 degree of freedom)
critical_value = 1.1

# Identify the bounds of the confidence interval
lower_bound_actual = Bo_values[np.where(likelihood_ratios <= critical_value)[0][0]]
upper_bound_actual = Bo_values[np.where(likelihood_ratios <= critical_value)[0][-1]]

# Create DataFrame for plotting
df_actual = pd.DataFrame({
    'Bo': Bo_values,
    'Objective Function Value': objective_values,
    'Profile Likelihood': profile_likelihood_values,
    'Normalized Likelihood': normalized_likelihoods,
    'Likelihood Ratio': likelihood_ratios
})

# Plot the data and the critical value line
plt.figure(figsize=(10, 6))
plt.plot(df_actual['Bo'], df_actual['Likelihood Ratio'], 'o', label='Likelihood Ratio')
plt.axhline(y=critical_value, color='r', linestyle='--', label=f'Critical Value = {critical_value}')
plt.axvline(x=lower_bound_actual, color='g', linestyle='--', label=f'Lower Bound = {lower_bound_actual}')
plt.axvline(x=upper_bound_actual, color='b', linestyle='--', label=f'Upper Bound = {upper_bound_actual}')
plt.xlabel('Parameter Bo [-]')
plt.ylabel('Likelihood Ratio [-]')
plt.title('Profile Likelihood with Rigorous Confidence Interval')
plt.legend()
plt.grid(True)
plt.show()

(lower_bound_actual, upper_bound_actual)
