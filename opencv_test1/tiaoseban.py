import cv2
import numpy as np
import pytesseract
img = cv2.imread("/home/python/Desktop/opencv_test/opencv_test1/gray_card_img1.jpg", 1)
print(pytesseract.image_to_string(img, lang="chi_sim+eng"))
# cv2.imshow("img1", img1)
# cv2.waitKey(0)