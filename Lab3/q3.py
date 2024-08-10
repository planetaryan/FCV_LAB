import numpy as np
import cv2

def box_filter(image, size):
    """Apply a box filter (averaging filter) to the image using cv2.filter2D."""
    # Create the box filter kernel
    kernel = np.ones((size, size), dtype=np.float32) / (size * size)

    # Apply the box filter using cv2.filter2D
    filtered_image = cv2.filter2D(image, -1, kernel)

    return filtered_image

def gaussian_kernel(size, sigma):
    """Generate a Gaussian kernel."""
    kernel = np.fromfunction(
        lambda x, y: (1/ (2 * np.pi * sigma**2)) *
                     np.exp(-((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma**2)),
        (size, size)
    )
    return kernel / np.sum(kernel)  # Normalize the kernel


# Parameters for Gaussian filter
kernel_size = 5  # Size of the Gaussian kernel
sigma = 1.0      # Standard deviation of the Gaussian distribution

# Load the image using OpenCV
original_image = cv2.imread('/home/student/PycharmProjects/220962408_CV/Lab3/images/zebra.jpeg', cv2.IMREAD_UNCHANGED)

# Check if the image has multiple channels and convert to grayscale if necessary
if len(original_image.shape) == 3:
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
else:
    gray_image = original_image

# Generate Gaussian kernel
gaussian_kernel = gaussian_kernel(kernel_size, sigma)

# Apply Gaussian filter
gaussian_filtered_image = cv2.filter2D(gray_image,-1, gaussian_kernel)

# Normalize Gaussian filtered image to match the image type (convert to uint8)
gaussian_filtered_image = np.uint8(np.clip(gaussian_filtered_image, 0, 255))

# Save the Gaussian filtered image
cv2.imwrite('images/gaussian_filtered_zebra.jpg', gaussian_filtered_image)

# Apply box filter
box_filtered_image = box_filter(gray_image, 3)

# Normalize box filtered image to match the image type (convert to uint8)
box_filtered_image = np.uint8(np.clip(box_filtered_image, 0, 255))

# Save the box filtered image
cv2.imwrite('images/box_filtered_zebra.jpg', box_filtered_image)