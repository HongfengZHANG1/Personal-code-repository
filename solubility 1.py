import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.optimize import minimize

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

# Define a function to calculate the Gaussian kernel matrix
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

# Define the objective function for optimization
def objective_function(sigma):
    # Compute the kernel matrix for training set
    K_train = gaussian_kernel_matrix(X_train, X_train, sigma)
    
    # Regularization term
    num_train = K_train.shape[0]
    alpha_train = lamda * np.identity(num_train)
    
    # Solve the linear problem using numpy.linalg.lstsq()
    c_train, _, _, _ = np.linalg.lstsq(K_train + alpha_train, Y_train, rcond=None)
    
    # Compute the kernel matrix for test set
    K_test_pred = gaussian_kernel_matrix(X_test, X_train, sigma)
    
    # Predict using the learned coefficients
    Y_pred = np.dot(K_test_pred, c_train)
    
    # Calculate the goodness of fit
    goodness_of_fit = np.linalg.norm(np.dot(K_test_pred, c_train) - Y_test)**2
    
    # Compute regularization term
    c_t = np.transpose(c_train)
    result_matrix = np.matmul(c_t, K_test_pred)
    regularization = np.matmul(result_matrix, c_train)
    
    # Objective function
    J = goodness_of_fit + regularization
    
    return J

# Hyperparameters
lamda = 1  # Regularization parameter

x0 = np.std(X_test, axis=0)
# Minimize the objective function to find the optimal sigma
result = minimize(objective_function, x0, method='Nelder-Mead', bounds=[(0.001, 1)] * len(x0))

# Extract the optimized sigma
optimized_sigma = result.x
print("Optimized sigma:", optimized_sigma)

