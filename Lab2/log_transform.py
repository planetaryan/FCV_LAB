import cv2
import numpy as np

img = cv2.imread('/home/student/PycharmProjects/220962408_CV/Lab2/images/flower.jpg')
img = img.astype(float)
c = 255/(np.log(1+np.max(img)))
log_transformed=c*np.log(1+img)

log_transformed = np.array(log_transformed, dtype=np.uint8)


cv2.imwrite('/home/student/PycharmProjects/220962408_CV/Lab2/images/flower_log_transformed.jpeg', log_transformed)
