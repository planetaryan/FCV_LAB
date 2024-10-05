import numpy as np
from sklearn.neighbors import NearestNeighbors
import cv2

def ratio_test(descriptors1, descriptors2, ratio_threshold=0.75):
    # Step 1: Use Nearest Neighbors to find the closest descriptors
    nbrs = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(descriptors2)
    distances, indices = nbrs.kneighbors(descriptors1)

    # Step 2: Apply the ratio test
    reliable_matches = []
    for i in range(len(descriptors1)):
        d1 = distances[i][0]  # Distance to the nearest neighbor
        d2 = distances[i][1]  # Distance to the second nearest neighbor

        # Prevent division by zero
        if d2 != 0:
            ratio = d1 / d2

            # Check if the ratio passes the threshold
            if ratio < ratio_threshold:
                # Store the index of the descriptor and its match
                reliable_matches.append((i, indices[i][0]))

    return reliable_matches

def main():
    # Load images (ensure the images exist in the specified paths)
    img1 = cv2.imread('images/tower2.jpg', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('images/tower3.jpg', cv2.IMREAD_GRAYSCALE)

    # Step 1: Detect ORB keypoints and compute descriptors
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    # Step 2: Apply the ratio test to filter matches
    matches = ratio_test(descriptors1, descriptors2)

    # Output results
    print("Reliable matches:", matches)
    print(f"Total matches found: {len(matches)}")

    # Optionally visualize the matches
    matched_image = cv2.drawMatches(img1, keypoints1, img2, keypoints2,
                                     [cv2.DMatch(m[0], m[1], 0) for m in matches],
                                     None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow("Matched Keypoints", matched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
