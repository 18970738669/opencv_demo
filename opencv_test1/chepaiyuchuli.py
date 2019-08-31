import cv2
import numpy as np

img = cv2.imread("/home/python/Desktop/opencv_test/opencv_test1/tupian.jpg")  # 读取图片
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换了灰度化
# cv2.imshow('gray', img_gray)  # 显示图片
# cv2.waitKey(0)
# 2、将灰度图像二值化，设定阈值是100
# img_thre = img_gray
# 灰点  白点 加错
# cv2.threshold(img_gray, 130, 255, cv2.THRESH_BINARY_INV, img_thre)
blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
ret3, img_thre1 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# cv2.imshow('threshold', img_thre)
# cv2.imwrite('wb_img.jpg', img_thre)
# cv2.waitKey(0)
# src=cv2.imread("/home/python/Desktop/opencv_test/opencv_test1/wb_img.jpg")
# src = img_thre
# src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
# height, width, channels = src.shape
# print("width:%s,height:%s,channels:%s" % (width, height, channels))
# for row in range(height):
#     for list in range(width):
#         for c in range(channels):
#             pv = src[row, list, c]
#             src[row, list, c] = 255 - pv
# cv2.imshow("AfterDeal", src)
img_thre = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
img_thre2 = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imshow("img_thre2", img_thre2)
cv2.imshow("img_thre", img_thre)
cv2.imshow("img_thre1", img_thre1)
# cv2.imwrite("img_thre1.jpg", img_thre1)
cv2.waitKey(0)
