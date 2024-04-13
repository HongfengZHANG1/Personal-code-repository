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
with open('C:/Users/12205/Desktop/Lent lecture/molecular modeling/Code and files/lmp170 (2).forces', 'r') as file:
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
skip_frames = 100

# Read the trajectory starting from the frame after skip_frames
for i, a in enumerate(ase.io.read("C:/Users/12205/Desktop/Lent lecture/molecular modeling/Code and files/lmp170 (3).xyz", index=":")):
    if i >= 100:
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

f1 = f[:501]
f2 = f[500:]

# Calculate autocorrelation function of forces for both halves
nlags = 200  # Adjust this value based on your preference
lag_acf1 = acf(f1, fft=True, nlags=nlags)
lag_acf2 = acf(f2, fft=True, nlags=nlags)


# Plot the autocorrelation function for both halves on the same plot
plt.figure(figsize=(10, 6), dpi=100)

plt.plot(range(len(lag_acf1)), lag_acf1, linestyle='-', linewidth=2, label='First Half')
plt.plot(range(len(lag_acf2)), lag_acf2, linestyle='-', linewidth=2, label='Second Half')

plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Function of Forces')
plt.axhline(y=0, color='black', linestyle='-', linewidth=1)  # Draw horizontal line at y=0
plt.legend()

plt.show()

plt.plot(range(len(f)), f, linestyle='-', linewidth=2)
plt.xlabel('Lag')
plt.ylabel('forces')
plt.title('time')
plt.axhline(y=0, color='black', linestyle='-', linewidth=1)  # Draw horizontal line at y=0
plt.show()

print(np.mean(f))