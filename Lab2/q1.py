import numpy as np
import cv2 as cv2

img = cv2.imread('/home/student/PycharmProjects/220962408_CV/Lab2/images/car_grayscale.jpeg')

hist, bins = np.histogram(img.flatten(), 256, [0, 256])

cdf = hist.cumsum()
cdf_normalized = cdf * float(hist.max()) / cdf.max()
cdf_m = np.ma.masked_equal(cdf,0)
cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
cdf = np.ma.filled(cdf_m,0).astype('uint8')

img2 = cdf[img]
cv2.imwrite('/home/student/PycharmProjects/220962408_CV/Lab2/images/car_hist_equalized.jpeg', img2)
