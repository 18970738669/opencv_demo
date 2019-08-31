import cv2
import numpy as np


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


def seperate_card(img, waves):
    part_cards = []
    for wave in waves:
        part_cards.append(img[:, wave[0]:wave[1]])
    return part_cards

def qiege(car_pic):
    color_pic = cv2.imread(car_pic, 1)
    img = cv2.imread(car_pic, 0)
    ret, gray_img = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    # cv2.imshow("gray_img", gray_img)
    # cv2.waitKey(0)
    # print("gray_img{}".format(gray_img))
    x_histogram = np.sum(gray_img, axis=1)
    # print(x_histogram)
    x_min = np.min(x_histogram)
    # print(x_histogram, x_min)
    x_average = np.sum(x_histogram) / x_histogram.shape[0]
    # print(x_average)
    x_threshold = (x_min + x_average) / 2
    wave_peaks = find_waves(x_threshold, x_histogram)
    # print(wave_peaks)
    # print("wavex:{}".format(wave_peaks))
    if len(wave_peaks) == 0:
        print("peak less 0:")
    # 认为水平方向，最大的波峰为车牌区域
    wave = max(wave_peaks, key=lambda x: x[1] - x[0])
    # print("wave:{}".format(wave))
    gray_img = gray_img[wave[0]:wave[1]]
    row_num, col_num = gray_img.shape[:2]
    # 去掉车牌上下边缘1个像素，避免白边影响阈值判断
    gray_img = gray_img[1:row_num - 1]
    y_histogram = np.sum(gray_img, axis=0)
    # print(y_histogram)
    y_min = np.min(y_histogram)
    y_average = np.sum(y_histogram) / y_histogram.shape[0]
    y_threshold = (y_min + y_average) / 5

    wave_peaks = find_waves(y_threshold, y_histogram)
    print("wavey:{}".format(wave_peaks))

