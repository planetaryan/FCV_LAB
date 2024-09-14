import cv2
import numpy as np
import matplotlib.pyplot as plt

def sift_custom(image):
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    if descriptors is None:
        return np.array([]), np.array([])
    return keypoints, descriptors

def calculate_pairwise_distances(desc1, desc2):
    diff = desc1[:, np.newaxis, :] - desc2[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=-1))
    return dist

def nearest_neighbour_matching(desc1, desc2):
    distances = calculate_pairwise_distances(desc1, desc2)
    matches = np.argmin(distances, axis=1)
    return matches

def visualize_keypoints(image, keypoints, title='Keypoints'):
    img_with_keypoints = cv2.drawKeypoints(image, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img_with_keypoints_rgb = cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 6))
    plt.imshow(img_with_keypoints_rgb)
    plt.title(title)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    image1 = cv2.imread('images/tower1.jpg', cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread('images/tower2.jpg', cv2.IMREAD_GRAYSCALE)
    keypoints1, desc1 = sift_custom(image1)
    keypoints2, desc2 = sift_custom(image2)
    if desc1.size == 0 or desc2.size == 0:
        print("No descriptors found in one of the images.")
    else:
        matches = nearest_neighbour_matching(desc1, desc2)
        print("Descriptor matches:", matches)
        visualize_keypoints(image1, keypoints1, 'Keypoints in Image 1')
        visualize_keypoints(image2, keypoints2, 'Keypoints in Image 2')
