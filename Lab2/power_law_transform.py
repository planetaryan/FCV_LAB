import cv2
import numpy as np

img=cv2.imread('/home/student/PycharmProjects/220962408_CV/Lab2/images/car.jpeg')
for gamma in [0.1,0.5,1.2,2.2]:
    gamma_corrected=np.array(255*(img/255)**gamma, dtype='uint8')

    cv2.imwrite('/home/student/PycharmProjects/220962408_CV/Lab2/images/gamma_transformed'+str(gamma)+'.jpeg',
                gamma_corrected)
