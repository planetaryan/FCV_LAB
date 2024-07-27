import cv2

img = cv2.imread('/home/student/PycharmProjects/220962408_CV/images/red_flower.jpeg')

new_width=1000
new_height=1000

resized_img=cv2.resize(img,(new_width,new_height))

cv2.imshow("Original", img)
cv2.imshow("Resized", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("/home/student/PycharmProjects/220962408_CV/images/resized_flower.jpeg", resized_img)

