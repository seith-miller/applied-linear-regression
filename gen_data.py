import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Save data to CSV file
data = np.hstack((X, y))  # Combine X and y into a single array
np.savetxt("synthetic_data.csv", data, delimiter=",", header="X,y", comments='')

# Plot the data
plt.scatter(X, y)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Synthetic Data')
plt.show()
