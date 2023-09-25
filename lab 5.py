import numpy as np
import matplotlib.pyplot as plt

def step_function(z):
  """Step activation function.

  Args:
    z: A real number.

  Returns:
    1 if z >= 0, else 0.
  """
  if z >= 0:
    return 1
  else:
    return 0

def perceptron(x, w):
  """Perceptron with step activation function.

  Args:
    x: A 1D numpy array representing the input features.
    w: A 1D numpy array representing the weights of the perceptron.

  Returns:
    The output of the perceptron.
  """
  z = np.dot(x, w)
  y = step_function(z)
  return y

def calculate_error(X, y, w):
  """Calculates the sum-square-error of the perceptron on the given training data.

  Args:
    X: A 2D numpy array representing the training input features.
    y: A 1D numpy array representing the training output labels.
    w: A 1D numpy array representing the weights of the perceptron.

  Returns:
    The sum-square-error of the perceptron.
  """
  error = 0
  for i in range(len(X)):
    x = X[i]
    t = y[i]

    y_pred = perceptron(x, w)

    error += (t - y_pred)**2

  return error

# Initial weights
w = np.array([10, 0.2, -0.75])

# Learning rate
alpha = 0.05

# Training data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])

# Error values
error_values = []

# Train the perceptron
for epoch in range(1000):
  for j in range(len(X)):
    x = X[j]
    t = y[j]

    # Calculate the output of the perceptron
    y_pred = perceptron(x, w)

    # Update the weights
    w += alpha * (t - y_pred) * x

  # Calculate the error
  error = calculate_error(X, y, w)

  # Append the error to the error values list
  error_values.append(error)

# Plot the epochs against the error values
plt.plot(range(len(error_values)), error_values)
plt.xlabel("Epoch")
plt.ylabel("Error")
plt.title("Error vs. Epochs for Perceptron")
plt.show()