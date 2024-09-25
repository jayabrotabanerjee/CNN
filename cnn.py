import numpy as np

def get_matrix_input(prompt):
    """Get matrix input from the user."""
    matrix = []
    print(prompt)
    while True:
        row = input("Enter a row (or blank to finish): ")
        if row == "":
            break
        matrix.append(list(map(int, row.strip())))
    return np.array(matrix)

def convolution(input_matrix, kernel, stride=1):
    """Perform convolution on the input matrix using the given kernel."""
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    output_height = (input_height - kernel_height) // stride + 1
    output_width = (input_width - kernel_width) // stride + 1
    output = np.zeros((output_height, output_width))

    for i in range(output_height):
        for j in range(output_width):
            output[i, j] = np.sum(input_matrix[i*stride:i*stride+kernel_height, j*stride:j*stride+kernel_width] * kernel)
    
    return output

def pooling(matrix, pool_type='max'):
    """Apply pooling operation to the matrix."""
    if pool_type == 'max':
        return np.max(matrix)
    elif pool_type == 'average':
        return np.mean(matrix)
    elif pool_type == 'sum':
        return np.sum(matrix)

# Input matrices
input_matrix = get_matrix_input("Enter the input matrix (each row on a new line):")
kernel = get_matrix_input("Enter the kernel matrix (each row on a new line):")

# Convolution with stride 1 and 2
conv_stride_1 = convolution(input_matrix, kernel, stride=1)
conv_stride_2 = convolution(input_matrix, kernel, stride=2)

print("Convolution Output (Stride 1):\n", conv_stride_1)
print("Convolution Output (Stride 2):\n", conv_stride_2)

# Pooling operations
pool_types = ['max', 'average', 'sum']
pooled_results = {}

# For 3x3 output from stride 1
for ptype in pool_types:
    pooled_results[f'3x3 {ptype}'] = pooling(conv_stride_1, pool_type=ptype)

# For 2x2 output from stride 2
for ptype in pool_types:
    pooled_results[f'2x2 {ptype}'] = pooling(conv_stride_2, pool_type=ptype)

# Print pooled results
for key, value in pooled_results.items():
    print(f"{key} Pooling Result: {value}")

# Flatten pooled feature maps
flattened_results = np.array(list(pooled_results.values()))
print("Flattened Pooled Feature Maps:\n", flattened_results)

# Simple learning simulation
weights = np.random.rand(flattened_results.size)
bias = 1

# Assuming a target value (for demonstration)
target = np.random.rand(1)[0]

# Learning over one epoch
learning_rate = 0.01
output = np.dot(weights, flattened_results) + bias
loss = target - output

# Update weights and bias
weights += learning_rate * loss * flattened_results
bias += learning_rate * loss

print("Updated Weights:\n", weights)
print("Updated Bias:\n", bias)
