# This code tests the parameters output by the user-written linear
# regression from linear_regression.py against the parameters
# output for linear regression over the same data by the np.polyfit
# function.

# Author: Mike Rushka
# Date created: 10/17/2025
# Date modified: 10/17/2025

import numpy as np
import pytest
from linear_regression_solutions import params, df

def test_1():

    # Fit data with linear regression using np.polyfit

    x = df['Age_years'].to_numpy(dtype=np.float64)
    y = df['Value_thousands_USD'].to_numpy(dtype=np.float64)

    alpha, beta = np.polyfit(x, y, 1)

    test_params = np.array([ alpha , beta ])

    assert np.allclose(params, test_params, atol = 1e-8), f"❌ Test failed: parameters do not match"


    print("✅ Test passed: Parameters are approximately equal.")