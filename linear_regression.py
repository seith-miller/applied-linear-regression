import numpy as np
import matplotlib.pyplot as plt
from utils import compute_cost

# Load the data
data = np.loadtxt("synthetic_data.csv", delimiter=",", skiprows=1)
X = data[:, 0].reshape(-1, 1)  # Extract X values
y = data[:, 1].reshape(-1, 1)  # Extract y values

# Add x0 = 1 to each instance (for the intercept term)
X_b = np.c_[np.ones((100, 1)), X]

# Compute the best-fit parameters using the normal equation
theta_best = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y

# Print the best-fit parameters
print(f"Intercept: {theta_best[0][0]}, Slope: {theta_best[1][0]}")

# Predict y values using the best-fit parameters
y_predict = X_b @ theta_best

# Compute the cost for the best-fit line
cost = compute_cost(X_b, y, theta_best)
print(f"Cost: {cost}")

# Plot the data and the regression line
plt.scatter(X, y, color="blue", label="Data")
plt.plot(X, y_predict, color="red", label="Best-fit line")
plt.xlabel('X')
plt.ylabel('y')
plt.title(f'Linear Regression Fit\nCost: {cost:.2f}')
plt.legend()
plt.show()
