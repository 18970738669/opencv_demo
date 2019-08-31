import cv2
import numpy as np

def qu_zao(img):
    kernel = np.ones((5, 5), np.float32) / 25
    res = cv2.filter2D(img, -1, kernel)
    # res2 = cv2.GaussianBlur(img, (5, 5), 0)
    # cv2.imshow('res', res)
    # cv2.imshow('res2', res2)
    # cv2.waitKey()
    return res



img = cv2.imread('/home/python/Desktop/opencv_test/pic/car4.jpg', 0)
kernel = np.ones((5, 5), np.float32) / 25
res = cv2.filter2D(img, -1, kernel)
res1 = cv2.blur(img, (5, 5))
res2 = cv2.GaussianBlur(img, (5, 5), 0)
res3 = cv2.medianBlur(img, 5)
res4 = cv2.bilateralFilter(img, 9, 75, 75)
cv2.imshow('pinghualvbo', res)
cv2.imshow('junzhilvbo', res1)
cv2.imshow('gaosilvbo', res2)
cv2.imshow('zhongzhilvbo', res3)
cv2.imshow('shuangbianlvbo', res4)
cv2.imshow('img', img)
cv2.waitKey()
