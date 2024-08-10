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

def compute_gradient_magnitude(grad_x, grad_y):
    """Compute the gradient magnitude."""
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return magnitude

def normalize_image(image):
    """Normalize image to range [0, 255]."""
    norm_image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(norm_image)

# Load a grayscale image
image = cv2.imread('/home/student/PycharmProjects/220962408_CV/Lab3/images/eiffel.jpg', cv2.IMREAD_GRAYSCALE)

# Ensure image is in float format
image = image.astype(np.float32)

# Compute Sobel gradients
grad_x, grad_y = compute_sobel_gradients(image)

# Compute the gradient magnitude
magnitude = compute_gradient_magnitude(grad_x, grad_y)

# Normalize the gradient magnitude image for display
magnitude_normalized = normalize_image(magnitude)

# Display results
plt.figure(figsize=(12, 6))
plt.title('Edge Magnitude')
plt.imshow(magnitude_normalized, cmap='gray')
plt.axis('off')

plt.show()
