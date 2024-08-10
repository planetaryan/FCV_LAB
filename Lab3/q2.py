import numpy as np
import cv2
import matplotlib.pyplot as plt

def compute_sobel_gradients(image):
    # Define Sobel kernels
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    sobel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=np.float32)

    # Compute gradients using cv2.filter2D
    grad_x = cv2.filter2D(image, cv2.CV_32F, sobel_x)
    grad_y = cv2.filter2D(image, cv2.CV_32F, sobel_y)

    return grad_x, grad_y

# Load a grayscale image
image = cv2.imread('/home/student/PycharmProjects/220962408_CV/Lab3/images/eiffel.jpg', cv2.IMREAD_GRAYSCALE)

# Ensure image is in float format
image = image.astype(np.float32)

# Compute Sobel gradients
grad_x, grad_y = compute_sobel_gradients(image)

# Display results
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.title('Gradient X')
plt.imshow(grad_x, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Gradient Y')
plt.imshow(grad_y, cmap='gray')
plt.axis('off')

plt.show()
