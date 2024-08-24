import numpy as np
import cv2

def gaussian_kernel(size, sigma=1):
    """Create a Gaussian kernel."""
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma**2)) * np.exp(
            - ((x - (size - 1) / 2)**2 + (y - (size - 1) / 2)**2) / (2 * sigma**2)
        ),
        (size, size)
    )
    return kernel / np.sum(kernel)

def apply_convolution(image, kernel):
    """Apply convolution with the given kernel using filter2D."""
    return cv2.filter2D(image, -1, kernel)

def sobel_gradients(image):
    """Compute gradients using Sobel operators."""
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    sobel_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ], dtype=np.float32)

    grad_x = apply_convolution(image, sobel_x)
    grad_y = apply_convolution(image, sobel_y)

    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    angle = np.arctan2(grad_y, grad_x) * (180.0 / np.pi) % 180

    return magnitude, angle

def non_max_suppression(magnitude, angle):
    """Apply non-maximum suppression to thin edges."""
    height, width = magnitude.shape
    suppressed = np.zeros_like(magnitude)

    angle = angle % 180
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            current_angle = angle[i, j]
            current_magnitude = magnitude[i, j]

            if (current_angle >= 0 and current_angle < 45) or (current_angle >= 135 and current_angle < 180):
                q1, q2 = magnitude[i, j + 1], magnitude[i, j - 1]
            elif (current_angle >= 45 and current_angle < 90):
                q1, q2 = magnitude[i + 1, j - 1], magnitude[i - 1, j + 1]
            elif (current_angle >= 90 and current_angle < 135):
                q1, q2 = magnitude[i - 1, j], magnitude[i + 1, j]

            if current_magnitude >= q1 and current_magnitude >= q2:
                suppressed[i, j] = current_magnitude

    return suppressed

def double_thresholding(image, low_threshold, high_threshold):
    """Apply double thresholding to classify edges."""
    strong = (image >= high_threshold)
    weak = (image >= low_threshold) & (image < high_threshold)
    return strong, weak

def edge_tracking_by_hysteresis(strong_edges, weak_edges):
    """Track edges by hysteresis."""
    height, width = strong_edges.shape
    output = np.zeros_like(strong_edges, dtype=np.uint8)

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            if strong_edges[i, j]:
                output[i, j] = 255
                for x in range(-1, 2):
                    for y in range(-1, 2):
                        if weak_edges[i + x, j + y]:
                            output[i + x, j + y] = 255

    return output

def canny_edge_detection(image, low_threshold=50, high_threshold=150):
    """Perform Canny edge detection."""
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Ensure the Gaussian kernel is correctly sized and sigma is appropriate
    gaussian = gaussian_kernel(size=5, sigma=1)
    smoothed = apply_convolution(gray_image, gaussian)

    magnitude, angle = sobel_gradients(smoothed)

    # Normalize magnitude to range [0, 255] for visualization
    magnitude = np.uint8(255 * magnitude / np.max(magnitude))

    suppressed = non_max_suppression(magnitude, angle)

    strong, weak = double_thresholding(suppressed, low_threshold, high_threshold)

    edges = edge_tracking_by_hysteresis(strong, weak)

    return edges

# Example usage:
if __name__ == "__main__":
    image = cv2.imread('images/zebra.jpeg')
    if image is None:
        print("Error: Image not found or unable to load.")
    else:
        edges = canny_edge_detection(image)
        cv2.imwrite('edges.jpg', edges)
        cv2.imshow('Edges', edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
