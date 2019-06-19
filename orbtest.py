import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('image/DJI_0171.JPG', 0)
img2 = cv2.imread('image/DJI_0173.JPG', 0)

# Initiate SIFT detector
orb = cv2.xfeatures2d.SIFT_create(0, 3, 0.04, 4)

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

good = []
for m,n in matches:
    if m.distance < 0.4 * n.distance:
        good.append([m])

# Draw first 10 matches.
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, 4372212, flags=2)

plt.imshow(img3)

plt.show()



