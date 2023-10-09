#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#A1


# In[8]:


import numpy as np
import matplotlib.pyplot as plt

# OR gate input 
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# OR gate output 
output_data = np.array([0, 0, 0, 1])

# Weights 
weight0, weight1, weight2 = 10, 0.2, -0.75
learning_rate = 0.05

# Step activation function
def custom_activation(z):
    return 1 if z >= 0 else 0

# Variables for tracking epochs and errors
epochs_count = 0
error_values = []

while True:
    total_error = 0
    for i in range(len(input_data)):
        input_point = input_data[i]
        target_output = output_data[i]
        
        # Compute the weighted sum
        weighted_sum = weight0 + weight1 * input_point[0] + weight2 * input_point[1]
        
        # Apply the step activation function
        predicted_output = custom_activation(weighted_sum)
        
        # Calculate the error
        error = target_output - predicted_output
        total_error += error ** 2
        
        # Update weights and bias
        weight0 += learning_rate * error
        weight1 += learning_rate * error * input_point[0]
        weight2 += learning_rate * error * input_point[1]
    
    epochs_count += 1
    error_values.append(total_error)
    
    # Check for convergence condition or maximum epochs
    if total_error <= 0.002 or epochs_count >= 1000:
        break

# Plot epochs vs. error values
plt.plot(range(epochs_count), error_values)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Training Progress for OR Gate Perceptron')
plt.grid(True)
plt.show()

# Print the learned weights and bias
print(f"Learned Weights: weight = {weight0}, weight1 = {weight1}, weight2 = {weight2}")


# In[ ]:


#A2


# In[10]:


import numpy as np
import matplotlib.pyplot as plt

# Define input data for an AND gate
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Define target output for an AND gate
target_output = np.array([0, 0, 0, 1])

# Initialize weights 
weight0, weight1, weight2 = 10, 0.2, -0.75
learning_rate = 0.05

# Define Bi-Polar Step activation function
def bi_polar_step_activation(z):
    return 1 if z > 0 else -1

# Initialize variables for tracking epochs and errors
epochs_count = 0
error_values = []

while True:
    total_error = 0
    for i in range(len(input_data)):
        input_point = input_data[i]
        target = target_output[i]
        
        # Calculate the weighted sum
        weighted_sum = weight0 + weight1 * input_point[0] + weight2 * input_point[1]
        
        # Apply the Bi-Polar Step activation function
        predicted_output = bi_polar_step_activation(weighted_sum)
        
        # Calculate the error
        error = target - predicted_output
        total_error += error ** 2
        
        # Update weights and bias
        weight0 += learning_rate * error
        weight1 += learning_rate * error * input_point[0]
        weight2 += learning_rate * error * input_point[1]
    
    epochs_count += 1
    error_values.append(total_error)
    
    # Check for convergence condition or maximum epochs
    if total_error <= 0.002 or epochs_count >= 1000:
        break

# Plot epochs vs. error values
plt.plot(range(epochs_count), error_values)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Epochs vs. Error for Bi-Polar Step Perceptron')
plt.grid(True)
plt.show()

# Print the learned weights and bias
print(f"Learned Weights: weight0 = {weight0}, weight1 = {weight1}, weight2 = {weight2}")


# In[11]:


import numpy as np
import matplotlib.pyplot as plt

# Define input data for an AND gate
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Define target output for an AND gate
target_output = np.array([0, 0, 0, 1])

# Initialize weights 
weight0, weight1, weight2 = 10, 0.2, -0.75
learning_rate = 0.05

# Define Sigmoid activation function
def sigmoid_activation(z):
    return 1 / (1 + np.exp(-z))

# Initialize variables for tracking epochs and errors
epochs_count = 0
errors = []

while True:
    total_error = 0
    for i in range(len(input_data)):
        input_point = input_data[i]
        target = target_output[i]
        
        # Calculate the weighted sum
        weighted_sum = weight0 + weight1 * input_point[0] + weight2 * input_point[1]
        
        # Apply the Sigmoid activation function
        predicted_output = sigmoid_activation(weighted_sum)
        
        # Calculate the error
        error = target - predicted_output
        total_error += error ** 2
        
        # Update weights and bias
        weight0 += learning_rate * error
        weight1 += learning_rate * error * input_point[0]
        weight2 += learning_rate * error * input_point[1]
    
    epochs_count += 1
    errors.append(total_error)
    
    # Check for convergence condition or maximum epochs
    if total_error <= 0.002 or epochs_count >= 1000:
        break

# Plot epochs vs. error values
plt.plot(range(epochs_count), errors)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Epochs vs. Error for Sigmoid Perceptron')
plt.grid(True)
plt.show()

# Print the learned weights and bias
print(f"Learned Weights: weight0 = {weight0}, weight1 = {weight1}, weight2 = {weight2}")


# In[12]:


import numpy as np
import matplotlib.pyplot as plt

# Define input data for an AND gate
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Define target output for an AND gate
target_output = np.array([0, 0, 0, 1])

# Initialize weights 
weight0, weight1, weight2 = 10, 0.2, -0.75
learning_rate = 0.05

# Define ReLU activation function
def relu_activation(z):
    return max(0, z)

# Initialize variables for tracking epochs and errors
epochs_count = 0
error_values = []

while True:
    total_error = 0
    for i in range(len(input_data)):
        input_point = input_data[i]
        target = target_output[i]
        
        # Calculate the weighted sum
        weighted_sum = weight0 + weight1 * input_point[0] + weight2 * input_point[1]
        
        # Apply the ReLU activation function
        predicted_output = relu_activation(weighted_sum)
        
        # Calculate the error
        error = target - predicted_output
        total_error += error ** 2
        
        # Update weights and bias
        weight0 += learning_rate * error
        weight1 += learning_rate * error * input_point[0]
        weight2 += learning_rate * error * input_point[1]
    
    epochs_count += 1
    error_values.append(total_error)
    
    # Check for convergence condition or maximum epochs
    if total_error <= 0.002 or epochs_count >= 1000:
        break

# Plot epochs vs. error values
plt.plot(range(epochs_count), error_values)
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.title('Epochs vs. Error for ReLU Perceptron')
plt.grid(True)
plt.show()

# Print the learned weights and bias
print(f"Learned Weights: weight0 = {weight0}, weight1 = {weight1}, weight2 = {weight2}")


# In[ ]:


#A3


# In[13]:


import numpy as np
import matplotlib.pyplot as plt

# Define input data for an AND gate
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Define target output for an AND gate
target_output = np.array([0, 0, 0, 1])

# Initialize weights 
initial_weight0, initial_weight1, initial_weight2 = 10, 0.2, -0.75

# Define Bi-Polar Step activation function
def bi_polar_step_activation(z):
    return 1 if z > 0 else -1

# List of learning rates to test
learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]

# A list to store the number of iterations needed for each learning rate
iterations_needed = []

for learning_rate in learning_rates:
    weight0, weight1, weight2 = initial_weight0, initial_weight1, initial_weight2
    
    # Variables for tracking epochs and errors
    epochs_count = 0
    
    while True:
        total_error = 0
        for i in range(len(input_data)):
            input_point = input_data[i]
            target = target_output[i]
            
            # Calculate the weighted sum
            weighted_sum = weight0 + weight1 * input_point[0] + weight2 * input_point[1]
            
            # Apply the Bi-Polar Step activation function
            predicted_output = bi_polar_step_activation(weighted_sum)
            
            # Calculate the error
            error = target - predicted_output
            total_error += error ** 2
            
            # Update weights and bias
            weight0 += learning_rate * error
            weight1 += learning_rate * error * input_point[0]
            weight2 += learning_rate * error * input_point[1]
        
        epochs_count += 1
        
        # Check for convergence condition or maximum epochs
        if total_error <= 0.002 or epochs_count >= 1000:
            break
    
    iterations_needed.append(epochs_count)

# Plot learning rates vs. iterations needed
plt.plot(learning_rates, iterations_needed, marker='o')
plt.xlabel('Learning Rate')
plt.ylabel('Iterations Needed for Convergence')
plt.title('Learning Rate vs. Iterations for Convergence')
plt.grid(True)
plt.show()

# Print the number of iterations needed for each learning rate
for i, rate in enumerate(learning_rates):
    print(f"Learning Rate {rate}: Iterations Needed = {iterations_needed[i]}")


# In[ ]:


#A4


# In[15]:


import numpy as np
import matplotlib.pyplot as plt

# Define input data for an XOR gate
input_data_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Define target output for an XOR gate
target_output_xor = np.array([0, 1, 1, 0])

# Initialize weights and bias
initial_weight0, initial_weight1, initial_weight2 = 10, 0.2, -0.75
learning_rate = 0.05

# Maximum number of epochs
max_epochs = 1000

# Lists to store epoch and error values
epoch_values = []
error_values = []

for epoch in range(max_epochs):
    total_error = 0.0
    for i in range(len(input_data_xor)):
        # Calculate the weighted sum
        weighted_sum = initial_weight0 + initial_weight1 * input_data_xor[i][0] + initial_weight2 * input_data_xor[i][1]
        
        # Apply Step activation function
        if weighted_sum > 0:
            predicted_output = 1
        else:
            predicted_output = 0
        
        error = target_output_xor[i] - predicted_output
        
        # Update weights and bias
        initial_weight0 += learning_rate * error
        initial_weight1 += learning_rate * error * input_data_xor[i][0]
        initial_weight2 += learning_rate * error * input_data_xor[i][1]
        
        # Add squared error to the total error
        total_error += error ** 2
    
    # Append epoch and error values for plotting
    epoch_values.append(epoch)
    error_values.append(total_error / len(input_data_xor))
    
    # Check for convergence
    if total_error / len(input_data_xor) <= 0.002:
        break

# Plot epochs against error values
plt.plot(epoch_values, error_values, marker='o', linestyle='-')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.title('Epochs vs. Error for XOR Gate')
plt.grid(True)
plt.show()

# Print the converged weights
print("Converged Weights:")
print("Weight0 =", initial_weight0)
print("Weight1 =", initial_weight1)
print("Weight2 =", initial_weight2)


# In[16]:


import numpy as np
import matplotlib.pyplot as plt

# Define input data for an XOR gate
input_data_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Define target output for an XOR gate
target_output_xor = np.array([0, 1, 1, 0])

# Initialize weights and bias
weight0, weight1, weight2 = 10, 0.2, -0.75

# Learning rate
learning_rate = 0.05

# Maximum number of epochs
max_epochs = 1000

# Lists to store epoch and error values
epoch_values = []
error_values = []

for epoch in range(max_epochs):
    total_error = 0.0
    for i in range(len(input_data_xor)):
        # Calculate the weighted sum
        weighted_sum = weight0 + weight1 * input_data_xor[i][0] + weight2 * input_data_xor[i][1]
        
        # Apply Bi-Polar Step activation function
        if weighted_sum > 0:
            predicted_output = 1
        else:
            predicted_output = -1
        
        # Calculate error
        error = target_output_xor[i] - predicted_output
        
        # Update weights and bias
        weight0 += learning_rate * error
        weight1 += learning_rate * error * input_data_xor[i][0]
        weight2 += learning_rate * error * input_data_xor[i][1]
        
        # Add squared error to the total error
        total_error += error ** 2
    
    # Append epoch and error values for plotting
    epoch_values.append(epoch)
    error_values.append(total_error / len(input_data_xor))
    
    # Check for convergence
    if total_error / len(input_data_xor) <= 0.002:
        break

# Print the number of epochs needed for convergence
print("Bi-Polar Step Activation Converged in", epoch + 1, "epochs")

# Plot epochs against error values
plt.plot(epoch_values, error_values)
plt.title('Epochs vs. Error (Bi-Polar Step Activation)')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()

# Print the converged weights
print("Converged Weights:")
print("Weight0 =", weight0)
print("Weight1 =", weight1)
print("Weight2 =", weight2)


# In[17]:


import numpy as np
import matplotlib.pyplot as plt

# Define input data for an XOR gate
input_data_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Define target output for an XOR gate
target_output_xor = np.array([0, 1, 1, 0])

# Initialize weights 
weight0, weight1, weight2 = 10, 0.2, -0.75

# Learning rate
learning_rate = 0.05

# Maximum number of epochs
max_epochs = 1000

# Lists to store epoch and error values
epoch_values = []
error_values = []

# Sigmoid activation function
def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

for epoch in range(max_epochs):
    total_error = 0.0
    for i in range(len(input_data_xor)):
        # Calculate the weighted sum
        weighted_sum = weight0 + weight1 * input_data_xor[i][0] + weight2 * input_data_xor[i][1]
        
        # Apply Sigmoid activation function
        predicted_output = sigmoid_activation(weighted_sum)
        
        # Calculate error
        error = target_output_xor[i] - predicted_output
        
        # Update weights and bias
        delta = learning_rate * error * predicted_output * (1 - predicted_output)
        weight0 += delta
        weight1 += delta * input_data_xor[i][0]
        weight2 += delta * input_data_xor[i][1]
        
        # Add squared error to the total error
        total_error += error ** 2
    
    # Append epoch and error values for plotting
    epoch_values.append(epoch)
    error_values.append(total_error / len(input_data_xor))
    
    # Check for convergence
    if total_error / len(input_data_xor) <= 0.002:
        break

# Print the number of epochs needed for convergence
print("Sigmoid Activation Converged in", epoch + 1, "epochs")

# Plot epochs against error values
plt.plot(epoch_values, error_values)
plt.title('Epochs vs. Error (Sigmoid Activation)')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()

# Print the converged weights
print("Converged Weights:")
print("Weight0 =", weight0)
print("Weight1 =", weight1)
print("Weight2 =", weight2)


# In[18]:


import numpy as np
import matplotlib.pyplot as plt

# Define input data for an XOR gate
input_data_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# Define target output for an XOR gate
target_output_xor = np.array([0, 1, 1, 0])

# Initialize weights 
weight0, weight1, weight2 = 10, 0.2, -0.75

# Learning rate
learning_rate = 0.05

# Maximum number of epochs
max_epochs = 1000

# Lists to store epoch and error values
epoch_values = []
error_values = []

# ReLU activation function
def relu_activation(x):
    return max(0, x)

for epoch in range(max_epochs):
    total_error = 0.0
    for i in range(len(input_data_xor)):
        # Calculate the weighted sum
        weighted_sum = weight0 + weight1 * input_data_xor[i][0] + weight2 * input_data_xor[i][1]
        
        # Apply ReLU activation function
        predicted_output = relu_activation(weighted_sum)
        
        # Calculate error
        error = target_output_xor[i] - predicted_output
        
        # Update weights and bias
        delta = learning_rate * error
        weight0 += delta
        weight1 += delta * input_data_xor[i][0]
        weight2 += delta * input_data_xor[i][1]
        
        # Add squared error to the total error
        total_error += error ** 2
    
    # Append epoch and error values for plotting
    epoch_values.append(epoch)
    error_values.append(total_error / len(input_data_xor))
    
    # Check for convergence
    if total_error / len(input_data_xor) <= 0.002:
        break

# Print the number of epochs needed for convergence
print("ReLU Activation Converged in", epoch + 1, "epochs")

# Plot epochs against error values
plt.plot(epoch_values, error_values)
plt.title('Epochs vs. Error (ReLU Activation)')
plt.xlabel('Epochs')
plt.ylabel('Error')
plt.show()

# Print the converged weights
print("Converged Weights:")
print("Weight0 =", weight0)
print("Weight1 =", weight1)
print("Weight2 =", weight2)


# In[ ]:


#A5


# In[19]:


import numpy as np

# Initialize weights and bias with random values
weight_candies, weight_mangoes, weight_milk_packets, bias_term = np.random.rand(4)

# Learning rate
learning_rate = 0.1

# Sigmoid activation function
def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

# Input features (Candies, Mangoes, Milk Packets)
transaction_data = np.array([
    [20, 6, 1],
    [16, 3, 2],
    [27, 9, 3],
    [19, 11, 0],
    [24, 8, 2],
    [15, 12, 1],
    [15, 4, 2],
    [18, 8, 2],
    [21, 1, 4],
    [24, 19, 8]
])

# Corresponding target labels (High Value Transaction)
target_labels = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0])

# Training the perceptron
for _ in range(1000):  # Adjust the number of epochs as needed
    total_error = 0
    for i in range(len(transaction_data)):
        # Compute the weighted sum of inputs
        weighted_sum = (
            weight_candies * transaction_data[i][0] +
            weight_mangoes * transaction_data[i][1] +
            weight_milk_packets * transaction_data[i][2] +
            bias_term
        )
        
        # Apply sigmoid activation function
        prediction = sigmoid_activation(weighted_sum)
        
        # Calculate the error
        error = target_labels[i] - prediction
        total_error += error ** 2
        
        # Update weights and bias
        weight_candies += learning_rate * error * prediction * (1 - prediction) * transaction_data[i][0]
        weight_mangoes += learning_rate * error * prediction * (1 - prediction) * transaction_data[i][1]
        weight_milk_packets += learning_rate * error * prediction * (1 - prediction) * transaction_data[i][2]
        bias_term += learning_rate * error
    
    # Check for convergence (adjust the error threshold as needed)
    if total_error < 0.01:
        break

# Classify new data point
def classify_transaction(candies, mangoes, milk_packets):
    weighted_sum = (
        weight_candies * candies +
        weight_mangoes * mangoes +
        weight_milk_packets * milk_packets +
        bias_term
    )
    prediction = sigmoid_activation(weighted_sum)
    return "High Value" if prediction >= 0.5 else "Low Value"

# Example 
new_transaction = [18, 7, 3]
classification_result = classify_transaction(*new_transaction)
print(f"New transaction {new_transaction} is classified as {classification_result}")


# In[ ]:


#A6


# In[20]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Input features (Candies, Mangoes, Milk Packets)
transaction_data = np.array([
    [20, 6, 1],
    [16, 3, 2],
    [27, 9, 3],
    [19, 11, 0],
    [24, 8, 2],
    [15, 12, 1],
    [15, 4, 2],
    [18, 8, 2],
    [21, 1, 4],
    [24, 19, 8]
])

# Corresponding target labels (High Value Transactions)
transaction_labels = np.array([1, 1, 1, 0, 1, 0, 1, 1, 0, 0])

# Create and train a custom perceptron model
class CustomPerceptron:
    def __init__(self, learning_rate=0.05, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            for i in range(n_samples):
                linear_output = np.dot(X[i], self.weights) + self.bias
                prediction = self.step_function(linear_output)

                # Update weights and bias
                self.weights += self.learning_rate * (y[i] - prediction) * X[i]
                self.bias += self.learning_rate * (y[i] - prediction)

    def step_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, X):
        predictions = []
        for i in range(X.shape[0]):
            linear_output = np.dot(X[i], self.weights) + self.bias
            predictions.append(self.step_function(linear_output))
        return np.array(predictions)

custom_perceptron = CustomPerceptron()
custom_perceptron.fit(transaction_data, transaction_labels)

# Create and train a logistic regression model
logistic_regression = LogisticRegression(solver='lbfgs')
logistic_regression.fit(transaction_data, transaction_labels)

# Generate predictions for each transaction
custom_perceptron_predictions = custom_perceptron.predict(transaction_data)
logistic_regression_predictions = logistic_regression.predict(transaction_data)

# Calculate accuracy for both models
custom_perceptron_accuracy = accuracy_score(transaction_labels, custom_perceptron_predictions)
logistic_regression_accuracy = accuracy_score(transaction_labels, logistic_regression_predictions)

# Plot transactions with actual and predicted labels
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(transaction_data[:, 0], transaction_data[:, 1], c=transaction_labels, cmap=plt.cm.Paired)
plt.title(f'Actual Labels (Custom Perceptron Accuracy: {custom_perceptron_accuracy:.2f})')

plt.subplot(1, 2, 2)
plt.scatter(transaction_data[:, 0], transaction_data[:, 1], c=custom_perceptron_predictions, cmap=plt.cm.Paired)
plt.title(f'Predicted Labels (Custom Perceptron Accuracy: {custom_perceptron_accuracy:.2f})')

plt.show()

# Print accuracy results
print("Custom Perceptron Accuracy:", custom_perceptron_accuracy)
print("Logistic Regression Accuracy:", logistic_regression_accuracy)


# In[ ]:


#A7


# In[21]:


import numpy as np

# AND gate input data
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# AND gate output data
output_labels = np.array([0, 0, 0, 1])

# Initialize weights and biases
np.random.seed(0)
input_size = 2
hidden_size = 2
output_size = 1

# Weights and biases for the input layer to the hidden layer
weights_input_to_hidden = np.random.uniform(size=(input_size, hidden_size))
biases_input_to_hidden = np.zeros(hidden_size)

# Weights and biases for the hidden layer to the output layer
weights_hidden_to_output = np.random.uniform(size=(hidden_size, output_size))
biases_hidden_to_output = np.zeros(output_size)

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Hyperparameters
learning_rate = 0.05
max_iterations = 1000
convergence_error = 0.002

# Training the neural network
for iteration in range(max_iterations):
    # Forward propagation
    hidden_layer_input = np.dot(input_data, weights_input_to_hidden) + biases_input_to_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_to_output) + biases_hidden_to_output
    output_layer_output = sigmoid(output_layer_input)

    # Calculate error
    error = output_labels.reshape(-1, 1) - output_layer_output

    # Backpropagation
    delta_output = error * sigmoid_derivative(output_layer_output)
    gradient_weights_hidden_to_output = np.dot(hidden_layer_output.T, delta_output)
    gradient_biases_hidden_to_output = np.sum(delta_output, axis=0)

    delta_hidden = np.dot(delta_output, weights_hidden_to_output.T) * sigmoid_derivative(hidden_layer_output)
    gradient_weights_input_to_hidden = np.dot(input_data.T, delta_hidden)
    gradient_biases_input_to_hidden = np.sum(delta_hidden, axis=0)

    # Update weights and biases
    weights_hidden_to_output += learning_rate * gradient_weights_hidden_to_output
    biases_hidden_to_output += learning_rate * gradient_biases_hidden_to_output
    weights_input_to_hidden += learning_rate * gradient_weights_input_to_hidden
    biases_input_to_hidden += learning_rate * gradient_biases_input_to_hidden

    # Calculate mean squared error
    mse = np.mean(np.square(error))

    # Check for convergence
    if mse <= convergence_error:
        print(f"Converged after {iteration + 1} iterations with MSE: {mse:.6f}")
        break

# Testing the trained neural network
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predicted_output = sigmoid(np.dot(sigmoid(np.dot(test_data, weights_input_to_hidden) + biases_input_to_hidden), weights_hidden_to_output) + biases_hidden_to_output)
predicted_labels = (predicted_output > 0.5).astype(int)

print("Test Data:")
print(test_data)
print("Predicted Labels:")
print(predicted_labels)


# In[ ]:


#A8


# In[22]:


import numpy as np

# XOR gate input data
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# XOR gate output data
output_labels = np.array([[0], [1], [1], [0]])

# Hyperparameters
learning_rate = 0.05
input_size = 2
hidden_size = 2
output_size = 1
max_iterations = 10000
convergence_error = 0.002

# Initialize weights with small random values
np.random.seed(0)
weights_input_to_hidden = 2 * np.random.random((input_size, hidden_size)) - 1
weights_hidden_to_output = 2 * np.random.random((hidden_size, output_size)) - 1

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Training the neural network using backpropagation
for iteration in range(max_iterations):
    # Forward propagation
    hidden_layer_input = np.dot(input_data, weights_input_to_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_to_output)
    output_layer_output = sigmoid(output_layer_input)

    # Calculate errors
    error_output = output_labels - output_layer_output
    error_output_delta = error_output * sigmoid_derivative(output_layer_output)

    error_hidden = error_output_delta.dot(weights_hidden_to_output.T)
    error_hidden_delta = error_hidden * sigmoid_derivative(hidden_layer_output)

    # Update weights
    weights_hidden_to_output += hidden_layer_output.T.dot(error_output_delta) * learning_rate
    weights_input_to_hidden += input_data.T.dot(error_hidden_delta) * learning_rate

    # Calculate mean squared error
    mse = np.mean(np.square(error_output))

    # Check for convergence
    if mse <= convergence_error:
        print(f"Converged after {iteration + 1} iterations with MSE: {mse:.6f}")
        break

# Testing the trained neural network
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predicted_output = sigmoid(np.dot(sigmoid(np.dot(test_data, weights_input_to_hidden)), weights_hidden_to_output))
predicted_labels = (predicted_output > 0.5).astype(int)

print("Test Data:")
print(test_data)
print("Predicted Labels:")
print(predicted_labels)


# In[ ]:


#A9


# In[23]:


import numpy as np

# XOR gate input data
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# XOR gate output data (two nodes for each output)
output_labels = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])

# Hyperparameters
learning_rate = 0.05
input_size = 2
hidden_size = 2
output_size = 2  # Two output nodes
max_iterations = 10000
convergence_error = 0.002

# Initialize weights with small random values
np.random.seed(0)
weights_input_to_hidden = 2 * np.random.random((input_size, hidden_size)) - 1
weights_hidden_to_output = 2 * np.random.random((hidden_size, output_size)) - 1

# Sigmoid activation function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Training the neural network using backpropagation
for iteration in range(max_iterations):
    # Forward propagation
    hidden_layer_input = np.dot(input_data, weights_input_to_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_to_output)
    output_layer_output = sigmoid(output_layer_input)

    # Calculate errors
    error_output = output_labels - output_layer_output
    error_output_delta = error_output * sigmoid_derivative(output_layer_output)

    error_hidden = error_output_delta.dot(weights_hidden_to_output.T)
    error_hidden_delta = error_hidden * sigmoid_derivative(hidden_layer_output)

    # Update weights
    weights_hidden_to_output += hidden_layer_output.T.dot(error_output_delta) * learning_rate
    weights_input_to_hidden += input_data.T.dot(error_hidden_delta) * learning_rate

    # Calculate mean squared error
    mse = np.mean(np.square(error_output))

    # Check for convergence
    if mse <= convergence_error:
        print(f"Converged after {iteration + 1} iterations with MSE: {mse:.6f}")
        break

# Testing the trained neural network
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predicted_output = sigmoid(np.dot(sigmoid(np.dot(test_data, weights_input_to_hidden)), weights_hidden_to_output))
predicted_labels = (predicted_output > 0.5).astype(int)

print("Test Data:")
print(test_data)
print("Predicted Labels:")
print(predicted_labels)


# In[ ]:


#A10


# In[24]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from matplotlib.colors import ListedColormap

# Define the AND gate input and output data
input_data_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
output_labels_and = np.array([0, 0, 0, 1])

# Define the XOR gate input and output data
input_data_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
output_labels_xor = np.array([0, 1, 1, 0])

# Create an MLPClassifier for the AND gate
and_classifier = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', solver='sgd', learning_rate_init=0.1, max_iter=10000)

# Create an MLPClassifier for the XOR gate
xor_classifier = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic', solver='sgd', learning_rate_init=0.1, max_iter=10000)

# Train the AND gate classifier
and_classifier.fit(input_data_and, output_labels_and)

# Train the XOR gate classifier
xor_classifier.fit(input_data_xor, output_labels_xor)

# Predict for AND gate inputs
and_predictions = and_classifier.predict(input_data_and)

# Predict for XOR gate inputs
xor_predictions = xor_classifier.predict(input_data_xor)

# Confusion matrix for AND gate
confusion_matrix_and = confusion_matrix(output_labels_and, and_predictions)

# Confusion matrix for XOR gate
confusion_matrix_xor = confusion_matrix(output_labels_xor, xor_predictions)

# Define a function to plot the decision boundary
def plot_decision_boundary(classifier, X, y, title):
    cmap_background = ListedColormap(['#FFAAAA', '#AAAAFF'])
    cmap_points = ListedColormap(['#FF0000', '#0000FF'])

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, cmap=cmap_background, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_points, marker='o')
    plt.title(title)
    plt.show()

# Plot decision boundary for AND gate
plot_decision_boundary(and_classifier, input_data_and, output_labels_and, "AND Gate Decision Boundary")

# Plot decision boundary for XOR gate
plot_decision_boundary(xor_classifier, input_data_xor, output_labels_xor, "XOR Gate Decision Boundary")

# Calculate accuracy for AND and XOR gates
accuracy_and = accuracy_score(output_labels_and, and_predictions)
accuracy_xor = accuracy_score(output_labels_xor, xor_predictions)

print("AND Gate Accuracy:", accuracy_and)
print("XOR Gate Accuracy:", accuracy_xor)


# In[ ]:


#A11


# In[3]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load data from the uploaded Excel file
data_df = pd.read_csv(r"C:\class\projects\sem 5\Machine Learning\Custom_CNN_Features.csv")

# Assuming 'text_column' contains the text data, replace it with the actual column name
text_column_name = 'f0'

# Drop rows with missing values in the text column
data_df = data_df.dropna(subset=[text_column_name])

# Convert the text column to strings
data_df[text_column_name] = data_df[text_column_name].astype(str)

# Convert 'Label' to categorical if needed
data_df['Label'] = data_df['Label'].astype('category')

# Split the data into training and testing sets
train_data, test_data = train_test_split(data_df, test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the training text data
X_train_tfidf = tfidf_vectorizer.fit_transform(train_data[text_column_name])

# Transform the testing text data
X_test_tfidf = tfidf_vectorizer.transform(test_data[text_column_name])

# Define the labels
y_train = train_data['Label']
y_test = test_data['Label']

# Create a decision tree classifier
tree_classifier = DecisionTreeClassifier()

# Train the classifier on the training data
tree_classifier.fit(X_train_tfidf, y_train)

# Make predictions on the testing data
y_pred = tree_classifier.predict(X_test_tfidf)

# Calculate accuracy
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy}")

# Plot the accuracy
plt.figure(figsize=(8, 6))
plt.bar(['Test Accuracy'], [test_accuracy], color='grey')
plt.ylim(0, 1)  # Set the y-axis limits between 0 and 1
plt.title('Classifier Accuracy')
plt.ylabel('Accuracy')
plt.show()

