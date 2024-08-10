import cv2
import numpy as np


def median_filter(image, size):
    """Apply median filter to the image."""
    pad_size = size // 2
    padded_image = np.pad(image, pad_size, mode='edge')
    filtered_image = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Extract the window
            window = padded_image[i:i + size, j:j + size]
            # Compute the median
            filtered_image[i, j] = np.median(window)

    return filtered_image

def laplacian_filter(image):
    """Apply Laplacian filter to the image using cv2.filter2D."""
    # Define Laplacian kernel
    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]], dtype=np.float32)
    # Apply Laplacian filter using cv2.filter2D
    laplacian_image = cv2.filter2D(image, cv2.CV_64F, kernel)
    return laplacian_image

# Load the image using OpenCV
original_image = cv2.imread('/home/student/PycharmProjects/220962408_CV/Lab3/images/eiffel.jpg', cv2.IMREAD_UNCHANGED)

# Check if the image has multiple channels and convert to grayscale if necessary
if len(original_image.shape) == 3:
    gray_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
else:
    gray_image = original_image

# Apply median filter
median_filtered = median_filter(gray_image, 3)

# Calculate the Laplacian of the median filtered image
laplacian = laplacian_filter(median_filtered)

# Normalize Laplacian to match the image type (convert to uint8)
laplacian = np.uint8(np.absolute(laplacian))

# Calculate the sharpened image
sharp_image = cv2.addWeighted(gray_image, 1, laplacian, -0.7, 0)

# Save the sharpened image
cv2.imwrite('images/sharpened_eiffel.jpg', sharp_image)

cv2.imshow('Sharpened Image', sharp_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
