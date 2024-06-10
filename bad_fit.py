import numpy as np
import matplotlib.pyplot as plt
from utils import compute_cost  # Import the compute_cost function

# Load the data
data = np.loadtxt("synthetic_data.csv", delimiter=",", skiprows=1)
X = data[:, 0].reshape(-1, 1)  # Extract X values
y = data[:, 1].reshape(-1, 1)  # Extract y values

# Add x0 = 1 to each instance (for the intercept term)
X_b = np.c_[np.ones((100, 1)), X]

# Generate random theta (intercept and slope)
theta_random = np.random.randn(2, 1)

# Compute the cost for the random line
cost = compute_cost(X_b, y, theta_random)

# Print the random theta and cost
print(f"Random Intercept: {theta_random[0][0]}, Random Slope: {theta_random[1][0]}")
print(f"Cost: {cost}")

# Predict y values using the random theta
y_predict_random = X_b @ theta_random

# Calculate the squared error for each point
squared_errors = np.square(y_predict_random - y)

# Plot the data and the random line
plt.scatter(X, y, color="blue", label="Data")
plt.plot(X, y_predict_random, color="green", label="Random best-fit line")

# Plot the squared errors for each point
for i in range(len(X)):
    plt.plot([X[i], X[i]], [y[i], y_predict_random[i]], 'r--')  # Vertical line to show error
    plt.text(X[i], (y[i] + y_predict_random[i]) / 2, f'{squared_errors[i][0]:.2f}', fontsize=8, color='red')

plt.xlabel('X')
plt.ylabel('y')
plt.title(f'Random Best-Fit Line\nTotal Cost: {cost:.2f}')
plt.legend()
plt.show()
