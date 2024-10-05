import cv2
import numpy as np

def find_homography(img1_path, img2_path):
    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Step 1: Detect ORB keypoints and compute descriptors
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    # Step 2: Match descriptors using the BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract location of good matches
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    # Step 3: Estimate homography using RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

    # Select inliers
    inliers = mask.ravel().tolist()
    good_matches = [matches[i] for i in range(len(inliers)) if inliers[i]]

    # Draw matches
    matched_image = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    return H, matched_image

def main():
    img1_path = 'images/tower2.jpg'  # Path to the first image
    img2_path = 'images/tower3.jpg'  # Path to the second image

    H, matched_image = find_homography(img1_path, img2_path)

    # Print the homography matrix
    print("Homography Matrix:\n", H)

    # Show the matched keypoints
    cv2.imshow("Matched Keypoints", matched_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
