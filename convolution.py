import numpy as np
import matplotlib.pyplot as plt

# Function for 2D convolution
def convolution2d(matrix, kernel, stride):
    m, n = matrix.shape
    k, l = kernel.shape
    output_shape = ((m - k) // stride + 1, (n - l) // stride + 1)
    convolved = np.zeros(output_shape)
    
    for i in range(0, m - k + 1, stride):
        for j in range(0, n - l + 1, stride):
            convolved[i // stride, j // stride] = np.sum(matrix[i:i+k, j:j+l] * kernel)
    
    return convolved

# Function to get input matrix and kernel from user
def get_matrix_input(prompt):
    matrix = []
    while True:
        row = input(prompt)
        if row.strip() == "":  # stop if empty input
            break
        matrix.append(list(map(float, row.split())))
    return np.array(matrix)

# Main program
if __name__ == "__main__":
    # Get user input for matrix
    print("Enter the input matrix (enter an empty line to finish):")
    matrix = get_matrix_input("Row (space-separated values): ")

    # Get user input for kernel
    print("Enter the kernel (enter an empty line to finish):")
    kernel = get_matrix_input("Row (space-separated values): ")

    # Perform convolution with stride 1
    conv_stride_1 = convolution2d(matrix, kernel, stride=1)
    print("Convolved Output:\n", conv_stride_1)

    # Visualize the convolved output
    plt.figure(figsize=(6, 6))
    plt.title("Convolved Output")
    plt.imshow(conv_stride_1, cmap='gray', aspect='auto')
    plt.colorbar()
    plt.axis('off')
    plt.show()

