import numpy as np
import matplotlib.pyplot as plt
import ase
from ase import io
from statsmodels.tsa.stattools import acf

E=[]

with open('C:/Users/12205/Desktop/Lent lecture/molecular modeling/Code and files/T 600/lmp180.log', 'r') as file:
    # Skip the first 781 lines
    for _ in range(781):
        next(file)

    # Read data from lines 782 to 10782
    for _ in range(10001):
        line = file.readline()
        values = [float(val) for val in line.split()]
        Energy = values[2]
        E.append(Energy)
               
f1 = E[0:5001]
f2 = E[5000:]

# Calculate autocorrelation function of forces for both halves
nlags = 200  # Adjust this value based on your preference
lag_acf1 = acf(f1, fft=True, nlags=nlags)
lag_acf2 = acf(f2, fft=True, nlags=nlags)


# Plot the autocorrelation function for both halves on the same plot
plt.figure(figsize=(10, 6), dpi=100)


plt.plot(range(len(lag_acf1)), lag_acf1, linestyle='-', linewidth=2, label='First Half')
plt.plot(range(len(lag_acf2)), lag_acf2, linestyle='-', linewidth=2, label='Second Half')

data=lag_acf2
def find_local_maxima(data, window_size=3):
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
    
def exponential_function(x):
    return np.exp(-x/33)   

# Generate x values
x_values = np.arange(0, 200, 1)

# Generate y values for the exponential function
y_exponential = exponential_function(x_values)
plt.plot(x_values, y_exponential, label=r'$ln(\tau)=e^{-\frac{\tau}{15}}$', linestyle='-', color='red')

lag_acf_sum1 = np.sum(lag_acf1[:50])
lag_acf_sum2 = np.sum(lag_acf2[:50])
tao_int1=0.5+lag_acf_sum1
tao_int2=0.5+lag_acf_sum2

print("Tao_int 1:", tao_int1)

print("Tao_int 2:", tao_int2)
# Exponential function
def exponential_function(x):
    return np.exp(-x/50)   

# Generate x values
x_values = np.arange(0, 200, 1)

# Generate y values for the exponential function
y_exponential = exponential_function(x_values)


plt.xlabel('Lag')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation Function of Energy')
plt.axhline(y=0, color='black', linestyle='-', linewidth=1)  # Draw horizontal line at y=0
plt.legend()

plt.show()
mean_value = np.mean(E)
print("Mean E:", mean_value)
variance = np.var(E)
print("Variance:", variance)

