import numpy as np
import cv2
from matplotlib import pyplot as plt

def k_means(X, K, max_iters=100, tol=1e-4):
    """
    K-means clustering algorithm.

    Parameters:
        X (numpy.ndarray): Data points, shape (n_samples, n_features).
        K (int): Number of clusters.
        max_iters (int): Maximum number of iterations.
        tol (float): Tolerance to declare convergence.

    Returns:
        tuple: (centroids, labels)
            centroids: Final cluster centers.
            labels: Cluster labels for each data point.
    """
    n_samples, n_features = X.shape

    # Randomly initialize centroids
    centroids = X[np.random.choice(n_samples, K, replace=False)]

    for _ in range(max_iters):
        # Compute distances from each point to each centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

        # Assign each point to the nearest centroid
        labels = np.argmin(distances, axis=1)

        # Compute new centroids
        new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(K)])

        # Check for convergence
        if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < tol):
            break

        centroids = new_centroids

    return centroids, labels

# Read image
img = cv2.imread('images/bricks.jpg')

# Convert it from BGR to RGB
img_RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Reshape image to an Mx3 array
img_data = img_RGB.reshape(-1, 3)

# Number of clusters
K = 5

# Perform K-means clustering
centroids, labels = k_means(img_data.astype(np.float32), K)

# Map each pixel to the centroid color
colours = centroids[labels].reshape(-1, 3)

# Reshape array to the original image shape
img_colours = colours.reshape(img_RGB.shape)

# Save the quantized image
output_path = 'images/quantized_image.png'
cv2.imwrite(output_path, cv2.cvtColor(img_colours.astype(np.uint8), cv2.COLOR_RGB2BGR))
print(f"Quantized image saved to {output_path}")

# Display the quantized image
plt.imshow(img_colours.astype(np.uint8))
plt.title('Quantized Image with K-means')
plt.axis('off')
plt.show()
