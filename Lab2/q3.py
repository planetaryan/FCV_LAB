import cv2

img = cv2.imread('/home/student/PycharmProjects/220962408_CV/Lab2/images/car.jpeg')
print(img.shape)
height,width,channels=img.shape

x1=80
x2=160
y1=150
y2=300
# Cropping an image
cropped_image = img[x1:x2, y1:y2]
new_width=1000
new_height=1000

resized_and_cropped_img=cv2.resize(cropped_image, (new_width,new_height))

cv2.imshow("Original", img)
cv2.imshow("Resized and Cropped", resized_and_cropped_img)
cv2.imwrite('/home/student/PycharmProjects/220962408_CV/Lab2/images/car_resized_cropped.jpeg',resized_and_cropped_img)


cv2.waitKey(0)
cv2.destroyAllWindows()