import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('/home/python/Desktop/opencv_test/pic/car7.jpg', 0)
plt.imshow(img, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])
plt.show()