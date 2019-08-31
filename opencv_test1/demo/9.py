import cv2
import numpy as np
import matplotlib as plt
img = cv2.imread("/home/python/Desktop/opencv_test/pic/car15.jpg", 0)
kernel = np.ones((20, 20), np.uint8)
img = cv2.GaussianBlur(img, (3, 3), 0)
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
img_opening = cv2.addWeighted(img, 1, opening, -1, 0)
ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
img_edge = cv2.Canny(img_thresh, 100, 200)
# 开运算
kernel = np.ones((4, 19), np.uint8)
img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)

image, contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 2000]
cv2.drawContours(img, contours, 0, (0, 0, 255), 3)
rect = cv2.minAreaRect(contours[0])
area_width, area_height = rect[1]
box = cv2.boxPoints(rect)
box = np.int0(box)
# print(area_height, area_width)

print('len(contours)', len(contours))
# cv2.imshow("kaiyunsuan", opening)
cv2.imshow("yuantu", img)
# cv2.imshow("img_opening", img_opening)
# cv2.imshow("img_thresh", img_thresh)
# cv2.imshow("img_edge2", img_edge2)
# cv2.imshow("img", img)
cv2.imshow("box", box)
cv2.waitKey(0)
cv2.destroyAllWindows()