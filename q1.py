import cv2

img = cv2.imread('/home/student/PycharmProjects/220962408_CV/images/red_flower.jpeg', 0)
cv2.imshow("Flower", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("/home/student/PycharmProjects/220962408_CV/images/gray_flower.jpeg", img)

