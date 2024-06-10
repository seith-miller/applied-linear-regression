import numpy as np
import matplotlib.pyplot as plt

# Load data from CSV file
data = np.loadtxt("synthetic_data.csv", delimiter=",", skiprows=1)
X = data[:, 0].reshape(-1, 1)  # Extract X values
y = data[:, 1].reshape(-1, 1)  # Extract y values

# Verify by plotting the data
plt.scatter(X, y)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Loaded Synthetic Data')
plt.show()
