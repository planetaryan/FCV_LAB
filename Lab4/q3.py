import cv2
import numpy as np

def segment_color(image_path, lower_color, upper_color, output_path):
    # Load the image
    image = cv2.imread(image_path)

    # Check if the image was loaded properly
    if image is None:
        print("Error: Unable to load image.")
        return

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper bounds for the color in HSV
    lower_bound = np.array(lower_color)
    upper_bound = np.array(upper_color)

    # Create a mask for the specified color
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Apply the mask to segment the color
    segmented_image = cv2.bitwise_and(image, image, mask=mask)

    # Save the result
    cv2.imwrite(output_path, segmented_image)
    print(f"Segmented image saved to {output_path}")

# Example usage
# Define the color range in HSV (Hue, Saturation, Value)
# These values are for a specific shade of blue
lower_color = [160, 50, 50]    # Lower bound in HSV
upper_color = [180, 255, 255]  # Upper bound in HSV

segment_color('images/car.jpeg', lower_color, upper_color, 'images/segmented_image.png')
