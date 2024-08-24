import cv2
import numpy as np
import math

def detect_edges(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    return edges

def hough_transform(edges, rho_res=1, theta_res=np.pi/180, threshold=100):
    # Get the dimensions of the image
    height, width = edges.shape
    diag_len = int(np.sqrt(height**2 + width**2))

    # Define the accumulator array
    rhos = np.arange(-diag_len, diag_len, rho_res)
    thetas = np.arange(-np.pi, np.pi, theta_res)
    accumulator = np.zeros((len(rhos), len(thetas)), dtype=int)

    # Edge points
    y_coords, x_coords = np.nonzero(edges)

    # Vote in the accumulator array
    for x, y in zip(x_coords, y_coords):
        for theta_idx, theta in enumerate(thetas):
            rho = int(x * np.cos(theta) + y * np.sin(theta))
            rho_idx = np.argmin(np.abs(rhos - rho))
            accumulator[rho_idx, theta_idx] += 1

    # Find lines in the accumulator array
    lines = []
    for rho_idx in range(accumulator.shape[0]):
        for theta_idx in range(accumulator.shape[1]):
            if accumulator[rho_idx, theta_idx] >= threshold:
                rho = rhos[rho_idx]
                theta = thetas[theta_idx]
                lines.append((rho, theta))

    return lines, accumulator

def draw_lines(image, lines):
    height, width = image.shape[:2]
    for rho, theta in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + width * (-b))
        y1 = int(y0 + height * (a))
        x2 = int(x0 - width * (-b))
        y2 = int(y0 - height * (a))
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return image

def main(image_path, output_path):
    image = cv2.imread(image_path)
    edges = detect_edges(image)
    lines, _ = hough_transform(edges)
    image_with_lines = draw_lines(image, lines)
    cv2.imwrite(output_path, image_with_lines)
    print(f"Lines detected and saved to {output_path}")

# Example usage
main('images/road2.jpeg', 'images/lines_detected.png')
