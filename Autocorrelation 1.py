import numpy as np
import matplotlib.pyplot as plt
import ase
from ase import io
from statsmodels.tsa.stattools import acf



# Initialize matrices
forces_1 = np.zeros(3)  # Assuming 3D forces
forces_2 = np.zeros(3)  # Assuming 3D forces

# List to store results
forces = []

# Read data from file
with open('C:/Users/12205/Desktop/Lent lecture/molecular modeling/Code and files/T 600/lmp180.forces', 'r') as file:
    # Skip the first line
    next(file)

    for line in file:
        # Split the line into values
        values = [float(val) for val in line.split()]

        # Extract the first three and next three values
        forces_1 = np.array(values[:3])
        forces_2 = np.array(values[3:6])

        # Append to matrices
        forces.append(forces_1 - forces_2)

# Read the trajectory file and calculate forces along the trajectory
dru = []  # List to store unit vectors
d = []  # List to store distances

# Skip the first 200 frames
skip_frames = 600

# Read the trajectory starting from the frame after skip_frames
for i, a in enumerate(ase.io.read("C:/Users/12205/Desktop/Lent lecture/molecular modeling/Code and files/T 600/lmp180.xyz", index=":")):
    if i >= 600:
        # Calculate relative displacement vector
        dr = (a.get_positions()[2, :] - a.get_positions()[642, :])
        # Calculate distance
        r = np.linalg.norm(dr)
        # Append distance to the list
        d.append(r)
        # Calculate unit vector
        dru1 = dr / np.linalg.norm(dr)
        dru.append(dru1)
        
    

# Calculate forces along the trajectory
f = []
for i in range(len(dru)):
    f.append((forces[i]) @ dru[i])

f1 = f[:5001]
f2 = f[5000:]

# Calculate autocorrelation function of forces for both halves
nlags = 300  # Adjust this value based on your preference
lag_acf1 = acf(f1, fft=True, nlags=nlags)
lag_acf2 = acf(f2, fft=True, nlags=nlags)


# Plot the autocorrelation function for both halves on the same plot
plt.figure(figsize=(10, 6), dpi=100)

plt.plot(range(len(lag_acf1)), lag_acf1, linestyle='-', linewidth=2, label='First Half')
plt.plot(range(len(lag_acf2)), lag_acf2, linestyle='-', linewidth=2, label='Second Half')


data=lag_acf2
def find_local_maxima(data, window_size=7):
    local_maxima = [(0,1)]

    for i in range(window_size, len(data) - window_size):
        if data[i] > max(data[i - window_size:i]) and data[i] > max(data[i + 1:i + window_size + 1]):
            local_maxima.append((i, data[i]))

    return local_maxima

# Example usage

local_maxima = find_local_maxima(data)


# Mark the local maxima on the plot
for index, value in local_maxima:
    plt.scatter(index, value, color='red', marker='o', s=20)
   

lag_acf_sum1 = np.sum(lag_acf1[:130])
lag_acf_sum2 = np.sum(lag_acf2[:130])
tao_int1=0.5+lag_acf_sum1
tao_int2=0.5+lag_acf_sum2
print("Tao_int 1:", tao_int1)

print("Tao_int 2:", tao_int2)
# Exponential function
def exponential_function(x):
    return np.exp(-x/55)   

# Generate x values
x_values = np.arange(0, 300, 1)

# Generate y values for the exponential function
y_exponential = exponential_function(x_values)
plt.plot(x_values, y_exponential, label=r'$e^{-\frac{\tau}{50}}$', linestyle='-', color='red')


plt.xlabel(r'$\tau$ (100 fs)')
plt.ylabel(r'$\rho(\tau)$')
plt.title('Autocorrelation Function of Forces')
plt.axhline(y=0, color='black', linestyle='-', linewidth=1)  # Draw horizontal line at y=0
plt.legend()
plt.show()


mean_value = np.mean(f)
print("Mean:", mean_value)
variance = np.var(f)
print("Variance:", variance)
