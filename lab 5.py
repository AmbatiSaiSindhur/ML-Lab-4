import numpy as np

# Define the step activation function
def step_activation(x):
    return 1 if x >= 0 else 0

# Define the perceptron function
def perceptron(input_data, weights):
    weighted_sum = np.dot(input_data, weights)
    output = step_activation(weighted_sum)
    return output

# Initialize the weights and learning rate
W0 = 10
W1 = 0.2
W2 = -0.75
learning_rate = 0.05

# Define the input data for AND gate logic
input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# Define the target output for AND gate logic
target_output = np.array([0, 0, 0, 1])

# Train the perceptron
converged = False
epochs = 0

while not converged:
    error_count = 0
    
    for i in range(len(input_data)):
        input_vector = np.insert(input_data[i], 0, 1) # Add bias term
        target = target_output[i]
        output = perceptron(input_vector, [W0, W1, W2])
        
        if output != target:
            error_count += 1
            W0 += learning_rate * (target - output) * input_vector[0]
            W1 += learning_rate * (target - output) * input_vector[[1]](#__1)
            W2 += learning_rate * (target - output) * input_vector[[2]](#__2)
    
    epochs += 1
    
    if error_count == 0 or epochs >= 1000:
        converged = True

# Print the final weights and number of epochs
print("Final Weights: W0 =", W0, "W1 =", W1, "W2 =", W2)
print("Number of Epochs:", epochs)
