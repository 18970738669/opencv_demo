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
    img = cv2.imread(car_pic, 1)
    # img = cv2.resize(img, (720, 180), interpolation=cv2.INTER_AREA)
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img1, (5, 5), 0)
    ret3, gray_img = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    # img2 = cv2.resize(img_thre, (200, 50), interpolation=cv2.INTER_AREA)
    # cv2.imshow("img", img)
    # cv2.imshow("img_thre", img_thre)
    # cv2.waitKey(0)
    # cv2.imshow("gray_img", gray_img)
    # cv2.waitKey(0)
    # print("gray_img{}".format(gray_img))
    x_histogram = np.sum(gray_img, axis=1)
    # print(x_histogram)
    x_min = np.min(x_histogram)
    # print(x_min)
    x_average = np.sum(x_histogram) / x_histogram.shape[0]
    # print(x_average)
    x_threshold = (x_min + x_average) / 2
    print("x_threshold:{}".format(x_threshold))
    wave_peaks = find_waves(x_threshold, x_histogram)
    # print(wave_peaks)
    print("wavex:{}".format(wave_peaks))
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
    y_min = np.min(y_histogram)
    y_average = np.sum(y_histogram) / y_histogram.shape[0]
    y_threshold = (y_min + y_average) / 5
    print("y_threshold:{}".format(y_threshold))
    wave_peaks = find_waves(y_threshold, y_histogram)
    print("wavey:{}".format(wave_peaks))

    # 车牌字符数应大于6
    if len(wave_peaks) <= 6:
        print("peak less 1:", len(wave_peaks))
    wave = max(wave_peaks, key=lambda x: x[1] - x[0])
    print("wave_max:{}".format(wave))
    max_wave_dis = wave[1] - wave[0]
    if len(wave_peaks) >= 10:       # 含有汉字川的情况
        if abs(wave_peaks[2][1] - wave_peaks[0][0] - max_wave_dis) <= 5:
            new_wave = (wave_peaks[0][0], wave_peaks[2][1])
            wave_peaks = wave_peaks[3:]
            wave_peaks.insert(0, new_wave)
            # print(wave_peaks)
    # 判断是否是左侧车牌边缘
    if wave_peaks[0][1] - wave_peaks[0][0] < max_wave_dis / 3 and wave_peaks[1][1] - wave_peaks[0][0]  > max_wave_dis:
        wave_peaks.pop(0)
    # 组合分离汉字
    cur_dis = 0
    for i, wave in enumerate(wave_peaks):
        if wave[1] - wave[0] + cur_dis > max_wave_dis * 0.5:
            break
        else:
            cur_dis += wave[1] - wave[0]
    if i > 0:
        wave = (wave_peaks[0][0], wave_peaks[i][1])
        wave_peaks = wave_peaks[i + 1:]
        wave_peaks.insert(0, wave)

    # 去除车牌上的分隔点
    point = wave_peaks[2]
    if point[1] - point[0] < max_wave_dis / 3:
        point_img = gray_img[:, point[0]:point[1]]
        if np.mean(point_img) < 255 / 5:
            wave_peaks.pop(2)

    if len(wave_peaks) <= 6:
        print("peak less 2:", len(wave_peaks))

    # cv2.imwrite("gray_card_img1.jpg", gray_img)
    # print("-------------------------")
    # print(gray_img)
    color_pic_cards = seperate_card(color_pic, wave_peaks)
    part_cards = seperate_card(gray_img, wave_peaks)
    # print(wave_peaks)
    # print(part_cards)

    return part_cards, color_pic_cards

part_cards, color_pic_cards = qiege("/home/python/Desktop/opencv_test/opencv_demo/opencv_test1/card_img_104_0.jpg")
for i, part_card in enumerate(part_cards):
    if np.mean(part_card) < 255 / 5:
        print("a point")
        continue
    # if i == 0:
    #     part_card = cv2.resize(part_card, (32, 40))
    part_card_old = part_card
    cv2.imwrite("qiegezifu1_{}.jpg".format(i), part_card)
    cv2.imshow("part_card2_{}".format(i), part_card)
    w = abs(part_card.shape[1] - 20) // 2
    part_card1 = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    part_card2 = cv2.resize(part_card1, (20, 20), interpolation=cv2.INTER_AREA)
    # if i ==0:
    #     print(pytesseract.image_to_string(part_card, lang="chi_sim"))
    # else:
    #     print(pytesseract.image_to_string(part_card, lang="eng"))

cv2.waitKey(0)