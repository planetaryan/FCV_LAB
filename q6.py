import cv2

# Load the image
img = cv2.imread('/home/student/PycharmProjects/220962408_CV/images/car.jpeg')

# Get the dimensions of the image
height, width, channels = img.shape

# Define the rotation angle (in degrees)
angle = 45

# Get the rotation matrix
rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1) #get center of image

# Rotate the image
rotated_img = cv2.warpAffine(img, rotation_matrix, (width, height))

print(f"Rotated image by {angle} degrees")

# Display the original and rotated images
cv2.imshow('Original', img)
cv2.imshow('Rotated', rotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
