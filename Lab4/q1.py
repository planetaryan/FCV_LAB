import cv2
import numpy as np
import os

def apply_thresholding(image_path, output_path_prefix):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image was loaded properly
    if image is None:
        print("Error: Unable to load image.")
        return

    # Define the output directory
    output_dir = '/home/student/PycharmProjects/220962408_CV/Lab4/images'

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Apply basic thresholding
    _, binary_basic = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(output_dir, f"{output_path_prefix}_basic_threshold.png"), binary_basic)

    # Apply inverse thresholding
    _, binary_inverse = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite(os.path.join(output_dir, f"{output_path_prefix}_inverse_threshold.png"), binary_inverse)

    # Apply adaptive thresholding
    binary_adaptive = cv2.adaptiveThreshold(image, 255,
                                            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY,
                                            11, 2)
    cv2.imwrite(os.path.join(output_dir, f"{output_path_prefix}_adaptive_threshold.png"), binary_adaptive)

    # Apply Otsu's thresholding
    _, binary_otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(os.path.join(output_dir, f"{output_path_prefix}_otsu_threshold.png"), binary_otsu)

    print("Thresholding results saved.")

# Example usage
apply_thresholding('images/car.jpeg', 'output_image')
