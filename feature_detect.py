import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('image/DJI_0173.JPG', 0)  

orb = cv2.xfeatures2d.SIFT_create(0, 3, 0.04, 4)
kp1, des1 = orb.detectAndCompute(img1, None)


temp_img = img1.copy()
temp_img = cv2.drawKeypoints(img1, kp1, temp_img, color=(0,255,0))
plt.imshow(temp_img)
plt.show()


