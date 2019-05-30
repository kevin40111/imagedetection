from matplotlib import pyplot as plt
import cv2 as cv

img1 = cv.imread('DJI_0002.JPG')
img2 = cv.imread('DJI_0003.JPG')
print('#### 1')

gray1 = img1
gray2 = img2
print('#### 2')

sift = cv.xfeatures2d.SIFT_create(200)
print('#### 3')

kp1, des1 = sift.detectAndCompute(gray1, None)
kp2, des2 = sift.detectAndCompute(gray1, None)
print('#### type 1: ', type(des1))
print('#### type 2: ', type(des2))
print('#### 4')

img1 = cv.drawKeypoints(gray1, kp1, img1)
img2 = cv.drawKeypoints(gray2, kp2, img2)

print('#### 5')

bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)
print('matches', len(matches))
print('#### 6')

good = [[m] for m, n in matches if m.distance < 0.5 * n.distance]

print('good', len(good))
print('#### 7')

img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
print('#### 8')

(r, g, b) = cv.split(img3)
img4 = cv.merge([r, g, b])
print('#### 9')

plt.imshow(img4)
plt.show()
