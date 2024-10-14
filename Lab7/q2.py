import cv2
import numpy as np
import glob

# Load your calibration results
mtx = np.array([[658.689006, 0, 302.55997536],
                 [0, 659.3762667, 244.33104757],
                 [0, 0, 1]])
dist = np.array([-2.49949338e-01, 9.01144719e-02, -1.37086510e-04, -2.14264602e-04, 6.61567604e-02])

# Define the checkerboard size
CHECKERBOARD = (12, 12)

# Prepare object points (3D points in real world space)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)

# Load images for reprojection
images = glob.glob('/home/student/Downloads/calib_example/*.tif')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,
                                             cv2.CALIB_CB_ADAPTIVE_THRESH +
                                             cv2.CALIB_CB_FAST_CHECK +
                                             cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret:
        # Refine the corner locations
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))

        # Get rotation and translation vectors for the current image
        success, rvec, tvec, inliers = cv2.solvePnPRansac(objp, corners2, mtx, dist)

        # Check if the function was successful
        if success:
            # Ensure rvec is a 1x3 array
            rvec = rvec.reshape((3,))

            # Project the 3D points back into the image plane
            projected_points, _ = cv2.projectPoints(objp, rvec, tvec, mtx, dist)

            # Draw the reprojected points on the original image
            for point in projected_points:
                cv2.circle(img, tuple(point[0].astype(int)), 5, (0, 255, 0), -1)  # Green dots for reprojected points

            # Draw the detected corners for reference
            for corner in corners2:
                cv2.circle(img, tuple(corner[0].astype(int)), 5, (255, 0, 0), -1)  # Blue dots for detected corners

            # Display the image with reprojected points
            cv2.imshow('Reprojection', img)
            cv2.waitKey(0)
        else:
            print(f"Reprojection failed for image: {fname}")
    else:
        print(f"Chessboard corners not found in image: {fname}")

cv2.destroyAllWindows()
