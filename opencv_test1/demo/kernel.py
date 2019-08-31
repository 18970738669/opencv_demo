import cv2
import numpy as np

img = cv2.imread("/home/python/Desktop/opencv_test/pic/car16.jpg", 0)

kernel = np.ones((5, 5), np.uint8)

erosion = cv2.erode(img, kernel, iterations=1)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
edges = cv2.Canny(img, 100, 200)
cv2.imshow("img", img)
# cv2.imshow("erosion", erosion)
# cv2.imshow("opening", opening)
# cv2.imshow("closing", closing)
cv2.imshow("edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()