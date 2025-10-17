# This code performs a basic linear regression on the car value data contained
# in the "car_value_vs_age.csv" file. Each of the data rows has only one response
# variable (car value in thousands of USD) and one predictor variable (car age
# in years). The standard gradient descent algorithm is used.

# Author: Mike Rushka
# Date created: 10/17/2025
# Date modified: 10/17/2025


# Set matplotlib backend to make sure it displays from the terminal
import matplotlib
matplotlib.use('TkAgg')

# Read in necessary packages
import pandas as pd
import numpy as np
import seaborn as sns # plotting package
import matplotlib.pyplot as plt # plotting package

# Read in car value data from csv file
df = pd.read_csv("car_value_vs_age.csv")

# Graph scatterplot of data
sns.set(style="whitegrid")
plt.figure(figsize=(8,6))
sns.scatterplot(x='Age_years', y='Value_thousands_USD', data=df, s=100, color='purple', edgecolor='black')
plt.title('Car Value vs Age')
plt.xlabel('Age (years)')
plt.ylabel('Value (thousands USD)')

# plt.show() # comment this in ONLY when you want to see the plot of the data, rest of the code won't run while plot is displaying

# Fit data

x = df['Age_years'].to_numpy(dtype=np.float64) # array of x values
y = df['Value_thousands_USD'].to_numpy(dtype=np.float64) # array of y values

# initial guesses for alpha and beta
alpha = 0.
beta = 0.

# learning rate and tolerance
eta = 1e-4
tol = 1e-6

params = np.array([alpha, beta]) # initial parameter vector, represents v_0

## DEFINE GRADIENT CALCULATING FUNCTION

def grad(x, y, alpha, beta):
    """

    Calculate gradient of sum of square errors (SSE) loss function

    Inputs: arrays of x and y values, scalar parameters alpha and beta
    Output: array containing partial derivatives of loss function w.r.t.
            linear regression parameters alpha and beta

    """
    errors = (alpha*x+beta-y)

    partial_alpha = np.dot(x, errors)
    partial_beta = np.sum(errors)

    return np.array([partial_alpha, partial_beta])


# perform gradient descent with (optional) iteration check

iter = 0
iter_limit = 1e6

##MAIN LOOP
while np.linalg.norm(grad(x, y, params[0], params[1])) > tol:

    params = params - eta * grad(x, y, params[0], params[1])
    iter += 1

    if iter > iter_limit:
        print("Iteration limit reached. Calculation failed.")
        break


print("Converged parameters:", params)


# Overlay regression line
x_vals = np.linspace(df['Age_years'].min(), df['Age_years'].max(), 100)  # fine grid for smooth line
y_vals = params[0] * x_vals + params[1]
plt.plot(x_vals, y_vals, color='red', linewidth=2, label=f'y = {params[0]:.2f}x + {params[1]:.2f}')

# Labels and title
plt.title('Car Value vs Age with Regression Line')
plt.xlabel('Age (years)')
plt.ylabel('Value (thousands USD)')
plt.legend()
# plt.show() COMMENT OUT if running pytest

