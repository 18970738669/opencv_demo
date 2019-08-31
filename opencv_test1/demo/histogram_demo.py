import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract


def find_waves(threshold, histogram):
    up_point = -1  # 上升点
    is_peak = False
    if histogram[0] > threshold:
        up_point = 0
        is_peak = True
    wave_peaks = []
    for i, x in enumerate(histogram):
        if is_peak and x < threshold:
            if i - up_point > 2:
                is_peak = False
                wave_peaks.append((up_point, i))
        elif not is_peak and x >= threshold:
            is_peak = True
            up_point = i
    if is_peak and up_point != -1 and i - up_point > 4:
        wave_peaks.append((up_point, i))
    return wave_peaks


color = "blue"
card_img = cv2.imread("/home/python/Desktop/opencv_test/pic/card_img.jpg", 1)
gray_img = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
ret, gray_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# print("gray_img{}".format(gray_img))
x_histogram = np.sum(gray_img, axis=1)
x_min = np.min(x_histogram)
# print(x_histogram, x_min)
x_average = np.sum(x_histogram) / x_histogram.shape[0]
x_threshold = (x_min + x_average) / 2
wave_peaks = find_waves(x_threshold, x_histogram)
print("wavex:{}".format(wave_peaks))
if len(wave_peaks) == 0:
    print("peak less 0:")
# 认为水平方向，最大的波峰为车牌区域
wave = max(wave_peaks, key=lambda x: x[1] - x[0])
gary_img = gray_img[wave[0]:wave[1]]
row_num, col_num = gray_img.shape[:2]
# 去掉车牌上下边缘1个像素，避免白边影响阈值判断
gray_img = gray_img[1:row_num - 1]
y_histogram = np.sum(gray_img, axis=0)
# print(y_histogram)
y_min = np.min(y_histogram)
y_average = np.sum(y_histogram) / y_histogram.shape[0]
y_threshold = (y_min + y_average) / 5  # U和0要求阈值偏小，否则U和0会被分成两半

wave_peaks = find_waves(y_threshold, y_histogram)
print("wavey:{}".format(wave_peaks))

# for wave in wave_peaks:
#	cv2.line(card_img, pt1=(wave[0], 5), pt2=(wave[1], 5), color=(0, 0, 255), thickness=2)
# 车牌字符数应大于6
if len(wave_peaks) <= 6:
    print("peak less 1:", len(wave_peaks))
wave = max(wave_peaks, key=lambda x: x[1] - x[0])
max_wave_dis = wave[1] - wave[0]
# 判断是否是左侧车牌边缘
if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis / 3 and wave_peaks[0][0] == 0:
    wave_peaks.pop(0)
# plt.imshow(x_gary_img), plt.title("x_gary_img")
# plt.show()
# cv2.imshow("x_gary_img", gary_img)
# cv2.imshow("img", gray_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
