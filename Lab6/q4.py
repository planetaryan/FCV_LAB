import cv2
import numpy as np

def stitch_images(img1, img2):
    # Step 1: Detect ORB keypoints and compute descriptors
    orb = cv2.ORB_create()
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    # Step 2: Match descriptors using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract location of good matches
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1, 2)

    # Step 3: Estimate homography using RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC)

    # Step 4: Warp img1 to img2's perspective
    height, width, _ = img2.shape
    img1_warped = cv2.warpPerspective(img1, H, (width, height))

    # Step 5: Create a panorama by combining the images
    # Create a blank canvas to hold the stitched result
    panorama = np.zeros((height, width, 3), dtype=np.uint8)

    # Place img2 on the canvas
    panorama[0:height, 0:width] = img2

    # Combine the warped img1 with img2
    for i in range(height):
        for j in range(width):
            if np.any(img1_warped[i, j]):
                panorama[i, j] = img1_warped[i, j]

    return panorama

def main():
    img1_path = 'images/tower2.jpg'  # Path to the first image
    img2_path = 'images/tower3.jpg'  # Path to the second image

    # Load images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # Stitch images
    panorama = stitch_images(img1, img2)

    # Show the result
    cv2.imshow("Panorama", panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
