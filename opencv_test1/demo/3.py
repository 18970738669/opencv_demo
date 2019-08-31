import cv2
import numpy as np

img = cv2.imread("/home/python/Desktop/opencv_test/opencv_test1/card_1.jpg")
img = cv2.GaussianBlur(img, (3, 3), 0)
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, img_thresh = cv2.threshold(img1, 0, 255, cv2.THRESH_OTSU)

cv2.imshow("img_thresh", img_thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
