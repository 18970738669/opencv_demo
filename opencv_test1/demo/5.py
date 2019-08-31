import cv2
import numpy as np
# 找出图像中的蓝色区域
img = cv2.imread("/home/python/Desktop/opencv_test/pic/car4.jpg", 1)
img2 = cv2.resize(img, (520, 520))

hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

low_blue = np.array([110, 50, 50])
upper_blue = np.array([130, 255, 255])

mask = cv2.inRange(hsv, low_blue, upper_blue)

res = cv2.bitwise_and(img2, img2, mask=mask)

# cv2.imshow('frame', frame)
cv2.imshow('mask', mask)
cv2.imshow('res', res)

cv2.waitKey(10000)



cv2.destroyAllWindows()
