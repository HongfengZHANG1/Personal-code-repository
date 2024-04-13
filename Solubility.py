import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Define a function to calculate the Gaussian kernel matrix
def gaussian_kernel_matrix(X1, X2, sigma):
    num1, num2 = X1.shape[0], X2.shape[0]
    K = np.zeros((num1, num2))
    for i in range(num1):
        for j in range(num2):
            product_kernel = 1.0
            for k in range(X1.shape[1]):
                product_kernel *= np.exp(-(X1[i, k] - X2[j, k]) ** 2 / (2 * sigma[k] ** 2))
                
            K[i, j] = product_kernel
    return K

# Function to calculate the mean squared error
def error_cal(Y_true, Y_pred):
    return np.sqrt(np.mean((Y_true - Y_pred)**2))

# Read the solubility dataset
sol = pd.read_csv("curated-solubility-dataset.csv")

# Extracting target solubility (Y) and defining a list of molecular properties (proplist)
Y = sol['Solubility']
proplist = ['HeavyAtomCount',
            'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms', 'NumRotatableBonds',
            'NumValenceElectrons', 'NumAromaticRings', 'NumSaturatedRings',
            'NumAliphaticRings', 'RingCount']

# Preprocessing the properties data (X)
X = np.array([list(sol[prop] / sol['MolWt']) for prop in proplist])
X = np.insert(X, 0, np.log(sol['MolWt']), axis=0)
X = X.T  # Transpose to have data points as rows and features as columns


# Divide X into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=42)

# Define the range of regularization parameters (sigma)
sigmaratio = [0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 1.7, 1.9]


# Initialize lists to store errors
error_train_lam = []
error_test_lam = []

# Loop over each value of lambda
for i in range(len(sigmaratio)):
    # Define the sigma parameter for the Gaussian kernel
    sigma1 = np.std(X_train, axis=0)*sigmaratio[i]  # You need to define an appropriate value for sigma
    lamda = 1
    # Compute the kernel matrix for training set
    K_train = gaussian_kernel_matrix(X_train, X_train, sigma1)

    # Regularization term
    alpha_train = lamda * np.identity(K_train.shape[0])

    # Solve the linear problem using numpy.linalg.lstsq()
    c_train = np.linalg.lstsq(K_train + alpha_train, Y_train, rcond=None)[0]

    # Compute the kernel matrix for test set
    K_test = gaussian_kernel_matrix(X_test, X_train, sigma1)

    # Predict using the learned coefficients
    Y_train_pred = np.dot(K_train, c_train)
    Y_test_pred = np.dot(K_test, c_train)
    
    # Calculate and store errors
    error_train_lam.append(error_cal(Y_train, Y_train_pred))
    error_test_lam.append(error_cal(Y_test, Y_test_pred))
    
    # Create a scatter plot of predicted vs. target solubility values for the test set
    
    plt.scatter(Y_test, Y_test_pred)
    plt.plot([-50, 50], [-50, 50], 'k--')
    plt.xlim(-12, 6)
    plt.ylim(-12, 6)
    plt.xlabel('Target Solubility')
    plt.ylabel('Predicted Solubility')
    plt.title(f" σ ratio = {sigmaratio[i]}")  # Add title with lambda value
    plt.gca().set_aspect('equal')
    plt.show()

# Print the errors for different values of lambda
print("Training set errors:", error_train_lam)
print("Test set errors:", error_test_lam)