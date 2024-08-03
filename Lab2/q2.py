import numpy as np
import cv2


def calculate_cdf(hist):
    """Calculate the cumulative distribution function (CDF) from a histogram."""
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]  # Normalize to 0-255
    return cdf_normalized


def histogram_specification(input_image, reference_image):
    # Read the input images
    img_input = cv2.imread(input_image, 0)
    img_reference = cv2.imread(reference_image, 0)

    # Calculate histograms
    hist_input, _ = np.histogram(img_input.flatten(), 256, [0, 256])
    hist_reference, _ = np.histogram(img_reference.flatten(), 256, [0, 256])

    # Calculate CDFs
    cdf_input = calculate_cdf(hist_input)
    cdf_reference = calculate_cdf(hist_reference)

    # Create mapping from input image CDF to reference image CDF
    mapping = np.interp(cdf_input, cdf_reference, np.arange(256))

    # Apply mapping to the input image
    img_mapped = np.interp(img_input.flatten(), np.arange(256), mapping).reshape(img_input.shape)

    # Convert mapped image to uint8 type
    img_mapped = img_mapped.astype(np.uint8)

    return img_mapped


# Define file paths
input_image_path = '/home/student/PycharmProjects/220962408_CV/Lab2/images/car_grayscale.jpeg'
reference_image_path = '/home/student/PycharmProjects/220962408_CV/Lab2/images/red_flower_grayscale.jpeg'
output_image_path = '/home/student/PycharmProjects/220962408_CV/Lab2/images/car_hist_spec.jpeg'

# Perform histogram specification
result_image = histogram_specification(input_image_path, reference_image_path)

# Save the result
cv2.imwrite(output_image_path, result_image)

