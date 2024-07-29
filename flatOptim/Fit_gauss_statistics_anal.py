from objects.Operator import Operator
import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import t
from scipy.optimize import leastsq

operator_instance = Operator()

#this parts loads experiment set from the folder, creates all the data handling objects and also does preprocessing

path = 'C:/UserData/z004d8nt/Documents/VSCHT/001_Paper_JChA/Paper1_Data/Suc_Glu_GE/'
experimentSet = operator_instance.Load_Experiment_Set(path)  #create object of the set of experiments
experimentCluster = operator_instance.Cluster_By_Component(experimentSet).clusters  # creates new object where components are clustered together
print('Starting preprocessing')
experimentSet_preprocessed = operator_instance.Preprocess(experimentSet, True, False, False, 0.005)  # does preprocessing
print('Preprocessing done')
experimentCluster_preprocessed = operator_instance.Cluster_By_Component(experimentSet_preprocessed).clusters  # creates new object where components are clustered together

'''object 'experimentCluster' is a dictionary of all the component names as a keys and experimentComp objects which hold
#chromatogram itself and information about the experiment setup, namely feed volume and concentration, column lenght,
#column diameter, flow rate, and commentary'''

i = 0
component_names_all = []
chromatograms = []
chromatograms_preprocessed = []
dead_volumes = []
concentrations = []
feed_volumes = []
feed_volumes_preprocessed = []
flow_rates = []
diameters = []
lengths = []
preprocessing_scores = []
inflection_widths = []

for comp_name, comp_objects in experimentCluster.items():
    # comp_objects is a list of ExperimentComponent objects
    for comp_object in comp_objects:  # Iterate over each ExperimentComponent object in the list
        chromatograms.append(comp_object.concentrationTime)  # Now you can access the concentrationTime attribute
        feed_volumes.append(comp_object.experiment.experimentCondition.feedVolume)
        dead_volumes.append(comp_object.experiment.experimentCondition.deadVolume)

for comp_name, comp_objects in experimentCluster_preprocessed.items():
    # comp_objects is a list of ExperimentComponent objects
    for comp_object in comp_objects:  # Iterate over each ExperimentComponent object in the list
        chromatograms_preprocessed.append(comp_object.concentrationTime)  # Now you can access the concentrationTime attribute
        component_names_all.append(comp_object.name)
        feed_volumes_preprocessed.append(comp_object.experiment.experimentCondition.feedVolume)
        concentrations.append(comp_object.feedConcentration)
        flow_rates.append(comp_object.experiment.experimentCondition.flowRate)
        diameters.append(comp_object.experiment.experimentCondition.columnDiameter)
        lengths.append(comp_object.experiment.experimentCondition.columnLength)
        preprocessing_scores.append(comp_object.preprocessingScore)
        inflection_widths.append(comp_object.inflectionWidth)
        i += 1

flow_rates_per_min = np.array(flow_rates) / 60
# Extracting the Time and A columns as numpy arrays
dead_times = np.array(dead_volumes) / flow_rates_per_min

chromatogram = chromatograms[12]

print('Number of available chromatograms: ' + str(len(chromatograms)))

# Shift the time to the left by dead_times[0]
chromatogram['Time'] -= dead_times[0]
# Remove rows with negative times
chromatogram = chromatogram[chromatogram['Time'] >= 0]
# Reset the index, if you want a nice consecutive index after dropping rows
chromatogram.reset_index(drop=True, inplace=True)

data_set = chromatogram.to_numpy()
data_set[:, 0] = data_set[:, 0]/60
max_time = data_set[-1, 0]
max_conc = max(data_set[:, 1])
max_conc_index = data_set[:, 1].tolist().index(max_conc)
# Initialize multiplier
multiplier = 1
# Define the scaling thresholds
scaling_factors = [5, 10, 100, 1000, 10000, 100000]
# Loop through scaling factors in reverse (to start with the largest)
for factor in reversed(scaling_factors):
   if max_conc < max_time / (factor * 10):
         multiplier = factor
         break
# Apply the multiplier to the dataset and max_conc
   data_set[:, 1] *= multiplier
   max_conc *= multiplier

# ---------------------Start of external code-------------------------------
def gaussian(x, a, b, c, d):
    amp = (a / (c * np.sqrt(2 * np.pi)))
    spread = np.exp((-(x - b) ** 2.0) / 2 * c ** 2.0)
    skew = (1 + erf((d * (x - b)) / (c * np.sqrt(2))))
    return amp * spread * skew

# defines the expected resultant as a sum of intrinsic gaussian functions
def GaussSum(x, p, n):
    gs = sum(gaussian(x, p[4*k], p[4*k+1], p[4*k+2], p[4*k+3])for k in range(n))
    return gs

# defines a residual, which is the  reducing the square of the difference
# between the data and the function
def residuals(p, y, x, n):
    return y - GaussSum(x, p, n)

# ---------------------End of external code---------------------------------------

init = data_set[max_conc_index, 0] + ((data_set[max_conc_index, 0]-data_set[max_conc_index-1, 0])/3)
initials = [[max_conc, init, 0.4, 0.0]]
n_value = len(initials)

'''#opt_params, covariance_matrix = curve_fit(gaussian, data_set[:, 0], data_set[:, 1], p0=initial_params)
opt_params, covariance_matrix, infodict, errmsg, ier = leastsq(residuals, initials, args=(data_set[:, 1], data_set[:, 0], n_value))[0]'''

# Perform the optimization
result = leastsq(residuals, initials, args=(data_set[:, 1], data_set[:, 0], n_value), full_output=True, maxfev=5000)

# Unpack the results
opt_params, cov_x, infodict, errmsg, ier = result

# Now you can check if a covariance matrix was calculated and handle it accordingly
if ier in [1, 2, 3, 4] and cov_x is not None:
    # Calculate the covariance matrix using the Jacobian returned in infodict
    # Note: To convert the Jacobian approximation to the covariance matrix, multiply it by the residuals variance
    # This assumes that the errors are normally distributed and the model is correct
    s_sq = (infodict['fvec']**2).sum()/(len(data_set[:, 1])-len(initials))
    covariance_matrix = cov_x * s_sq
else:
    covariance_matrix = None
    print("Covariance matrix couldn't be computed:", errmsg)


print("Optimized Parameters [A, B, C, D]:", opt_params)

time = chromatogram['Time'].values
concentration = chromatogram.iloc[:,1].values

# Generate fitted values
fitted_concentration = gaussian(time, *opt_params)
fitted_concentration = fitted_concentration #/ multiplier

alpha = 0.05  # For 95% confidence intervals
n = len(concentration)  # Number of data points
p = len(opt_params)  # Number of parameters
dof = max(0, n - p)  # Degrees of freedom

# t.ppf is the inverse of the CDF (Cumulative Distribution Function)
tval = t.ppf(1.0-alpha/2., dof)

for i, param in enumerate(opt_params):
    sigma = np.sqrt(covariance_matrix[i, i])
    ci_low = param - sigma*tval
    ci_high = param + sigma*tval
    print(f"Parameter {i} = {param:.4f}, Confidence Interval: [{ci_low:.4f}, {ci_high:.4f}]")

num_samples = 1000
param_samples = np.random.normal(loc=opt_params, scale=np.sqrt(np.diag(covariance_matrix)), size=(num_samples, len(opt_params)))
output_samples = np.array([gaussian(time, *params) for params in param_samples])

# Calculate the percentile of outputs to get the confidence area
lower_percentile = np.percentile(output_samples, 2.5, axis=0)
upper_percentile = np.percentile(output_samples, 97.5, axis=0)

'''plt.figure(figsize=(10, 6))
plt.plot(time, concentration, 'b-', label='Original Data')
plt.plot(time, fitted_concentration, 'r--', label='Fitted Curve')
plt.fill_between(time, lower_percentile, upper_percentile, color='gray', alpha=0.2, label='95% Confidence Area')
plt.legend()
plt.xlabel('Time (min)')
plt.ylabel('Concentration (scaled)')
plt.title('Fitted Curve with 95% Confidence Area')
plt.show()'''

# Set the font to Times New Roman and increase font size for better readability
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 14  # Adjust the font size as needed

# Create a figure and axis with a specific size suitable for fitting three plots on an A4 sheet
plt.figure(figsize=(6, 4))

# Plot the original data as red squares
plt.plot(time, concentration, 'rs', label='Original Data')

# Plot the fitted curve as a solid blue line
plt.plot(time, fitted_concentration, 'b-', linewidth=2, label='Fitted Curve')

# Fill the area between the lower and upper percentiles
plt.fill_between(time, lower_percentile, upper_percentile, color='gray', alpha=0.2, label='95% Confidence Area')

# Add a legend with best location
#plt.legend(loc='best', fontsize=12)  # Increase legend font size

# Remove the grid
plt.grid(False)

# Label the axes
plt.xlabel('Time (min)', fontsize=16)  # Increase label font size
plt.ylabel('Concentration [g/L]', fontsize=16)  # Increase label font size

# Increase the size of tick labels
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Set the limits for axes to start at 0 and end at 180 for the x-axis
plt.xlim(0, 180)
plt.ylim(min(lower_percentile), max(upper_percentile))

# Display the plot
plt.show()

# Calculate deviations
deviations = concentration - fitted_concentration

# Calculate standard deviation of the deviations
std_deviation = np.std(deviations)
print('Standard deviation ' + str(std_deviation))

norm_std_deviation = std_deviation / (max_conc/multiplier)
print('Normalized standard deviation ' + str(norm_std_deviation))

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)

# Scatter plot of original chromatogram data points
axs[0].scatter(time, concentration, c='blue', label='Original Data', s=20)  # s is the marker size
# Line plot of the fitted curve
axs[0].plot(time, fitted_concentration, 'r--', label='Fitted Curve')
axs[0].set_ylabel('Concentration')
axs[0].legend()
axs[0].set_title('Chromatogram and Fitted Curve')

# Line plot of deviations
axs[1].plot(time, deviations, 'b-', label='Deviations')
# Shaded area representing ±1 standard deviation
axs[1].fill_between(time, -std_deviation, std_deviation, color='red', alpha=0.2, label='±1 Std Deviation')
# Zero deviation line for reference
axs[1].axhline(y=0, color='k', linewidth=0.5)
axs[1].set_xlabel('Time (min)')
axs[1].set_ylabel('Deviation')
axs[1].legend()

plt.tight_layout()
plt.show()

# Assuming 'time' is your array of time values, 'A' is your array of observed data points,
# and 'fitted_A' is the array of values predicted by your fitted curve.

# Calculate residuals
residuals = concentration - fitted_concentration

# Plotting the residual plot
plt.figure(figsize=(10, 6))
plt.scatter(time, residuals, color='blue', label='Residuals', s=20)  # s is the marker size
plt.axhline(y=0, color='red', linestyle='--', label='Zero Error')
plt.xlabel('Time (min)')
plt.ylabel('Residuals (Observed - Predicted)')
plt.title('Residual Plot')
plt.legend()
plt.grid(True)
plt.show()
