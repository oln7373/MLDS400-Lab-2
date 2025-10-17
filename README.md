Hi everyone,

This is a demonstration repo showing how to set up a repo, upload new code to it, and test said code automatically using Github workflows. The files include:

linear_regression_solutions.py - Code which fits input data with a linear regression program written by me (via gradient descent).
lin_reg_tests.py - Code which tests the Mike-written linear regression against numpy's linear regression function.
requirements.txt - A list of packages needed to run all files in the repo.
main.yml - Located in the .github/workflows folder and contains all the instructions for setting up and running a virtual Python environment in your repo.
