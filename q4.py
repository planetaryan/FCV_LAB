import cv2

img = cv2.imread('/home/student/PycharmProjects/220962408_CV/images/red_flower.jpeg')

cv2.rectangle(img,(100,50),(200,100),(255,0,0),2) #top_left,bottom_right,color,thickness
cv2.imshow("Flower", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("/home/student/PycharmProjects/220962408_CV/images/rect_flower.jpeg", img)

