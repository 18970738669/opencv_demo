
import cv2
import numpy as np

img = cv2.imread("/home/python/Desktop/opencv_test/pic/img_edge2.jpg", 0)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()