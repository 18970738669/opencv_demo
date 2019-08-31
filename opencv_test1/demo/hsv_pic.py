import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("/home/python/Desktop/opencv_test/pic/car3.jpg", 1)
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
hsv_img_h = hsv_img[..., 0]
hsv_img_s = hsv_img[..., 1]
hsv_img_v = hsv_img[..., 2]
print(hsv_img.item(100, 100, 2))
print(hsv_img.item(100, 110, 2))

row, col = hsv_img.shape[:2]
print(row, col)
# plt.subplot(221)
# plt.imshow(hsv_img)
# plt.axis('off')
# plt.title('HSV')
#
# plt.subplot(222)
# plt.imshow(hsv_img_h, cmap='gray')
# plt.axis('off')
# plt.title('H')
#
# plt.subplot(223)
# plt.imshow(hsv_img_s, cmap='gray')
# plt.axis('off')
# plt.title('S')
#
# plt.subplot(224)
# plt.imshow(hsv_img_v, cmap='gray')
# plt.axis('off')
# plt.title('V')
# plt.show()

cv2.imshow("bsv_img", hsv_img)
cv2.waitKey(0)
cv2.destroyAllWindows()