import cv2
import numpy as np


SZ = 20


def point_limit(point):
    if point[0] < 0:
        point[0] = 0
    if point[1] < 0:
        point[1] = 0


def accurate_place(card_img_hsv, limit1, limit2, color):
    row_num, col_num = card_img_hsv.shape[:2]
    xl = col_num
    xr = 0
    yh = 0
    yl = row_num
    row_num_limit = 21
    col_num_limit = col_num * 0.8 if color != "green" else col_num * 0.5  # 绿色有渐变
    for i in range(row_num):
        count = 0
        for j in range(col_num):
            H = card_img_hsv.item(i, j, 0)
            S = card_img_hsv.item(i, j, 1)
            V = card_img_hsv.item(i, j, 2)
            if limit1 < H <= limit2 and 34 < S and 46 < V:
                count += 1
        if count > col_num_limit:
            if yl > i:
                yl = i
            if yh < i:
                yh = i
    for j in range(col_num):
        count = 0
        for i in range(row_num):
            H = card_img_hsv.item(i, j, 0)
            S = card_img_hsv.item(i, j, 1)
            V = card_img_hsv.item(i, j, 2)
            if limit1 < H <= limit2 and 34 < S and 46 < V:
                count += 1
        if count > row_num - row_num_limit:
            if xl > j:
                xl = j
            if xr < j:
                xr = j
    return xl, xr, yh, yl


def dingwei(car_pic):
    MAX_WIDTH = 2000
    # 先读取图片
    img = cv2.imread(car_pic, 1)
    # 取得照片的宽高
    pic_hight, pic_width = img.shape[0:2]
    print(pic_hight, pic_width)
    # 对照片大小进行调整
    if pic_width > MAX_WIDTH:
        resize_rate = MAX_WIDTH / pic_width
        img = cv2.resize(img, (MAX_WIDTH, int(pic_hight * resize_rate)), interpolation=cv2.INTER_AREA)
    # 开运算卷积核
    # img = cv2.resize(img, (600, 450), interpolation=cv2.INTER_AREA)
    oldimg = img
    kernel = np.ones((20, 20), np.uint8)
    # 高斯滤波去噪
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # 开运算去噪
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    # 增加白点数用于精准定位车牌位置
    img_opening = cv2.addWeighted(img, 1, opening, -1, 0)
    # 利用阀值把图片转换成二进制图片
    ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # ret, img_thresh = cv2.threshold(img_opening, 0, 255, cv2.THRESH_OTSU)
    # 找出图片边缘
    img_edge = cv2.Canny(img_thresh, 100, 200)
    # 使用开运算和闭运算让图像边缘成为一个整体img
    kernel = np.ones((4, 22), np.uint8)
    # kernel = np.ones((5, 12), np.uint8)
    # kernel = np.ones((4, 85), np.uint8)
    img_edge1 = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, kernel)
    img_edge2 = cv2.morphologyEx(img_edge1, cv2.MORPH_OPEN, kernel)
    # 找出图片边缘为矩形的轮廓(车牌就在这些轮廓图中)
    image, contours, hierarchy = cv2.findContours(img_edge2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1900]
    cv2.drawContours(img, contours, 0, (0, 0, 255), 3)
    # print(len(contours))
    # cv2.drawContours(img, contours, 1, (0, 0, 255), 3)
    # cv2.drawContours(img, contours, 2, (0, 0, 255), 3)
    # rect = cv2.minAreaRect(contours[0])
    # print("alpha=%d" % rect[2])
    # box = cv2.boxPoints(rect)
    # box = np.int0(box)opencv_test1/dingwei.py:272
    car_contours = []
    # oldimg = cv2.drawContours(oldimg, [box], 0, (0, 0, 255), 2)
    # cv2.imshow("oldimg", oldimg)opencv_test1/dingwei.py:100
    # cv2.imshow("edge4", oldimg)
    # cv2.imshow("opening", opening)
    # cv2.imshow("img_edge1", img_edge1)
    # cv2.imshow("img_thresh", img_thresh)
    # cv2.imshow("img_edge2.jpg", img_edge2)
    # cv2.imshow("img.jpg", img)
    # cv2.waitKey(0)
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)  # 返回值元组（（最小外接矩形的中心坐标），（宽，高），旋转角度）----->    ((x, y), (w, h), θ )
        area_width, area_height = rect[1]

        if area_width < area_height:
            area_width, area_height = area_height, area_width
        wh_ratio = area_width / area_height
        # print(wh_ratio)
        # 要求矩形区域长宽比在2到6之间，2到6是车牌的长宽比，其余的矩形排除
        if wh_ratio > 2 and wh_ratio < 6:
            car_contours.append(rect)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            # print(rect)
    print(len(car_contours))
    # print("cnt={}".format(cnt))
    card_imgs=[]
    for rect in car_contours:
        if rect[2] > -1 and rect[2] < 1:
            angle = 1
        else:
            angle = rect[2]
        rect = (rect[0], (rect[1][0] + 5, rect[1][1] + 5), angle)  # 扩大范围，避免车牌边缘被排除
        box = cv2.boxPoints(rect)
        # print("alpha={}".format(rect[2]))
        heigth_point = right_point = [0, 0]
        # left_point = low_point = rect[1]
        left_point = low_point = [pic_width, pic_hight]
        # print("pic_width:{}, pic_hight{}".format(pic_width, pic_hight))
        point_set = []
        for point in box:
            # print(point)
            # print(left_point)
            if left_point[0] > point[0]:
                left_point = point
            if low_point[1] > point[1]:
                low_point = point
            if heigth_point[1] < point[1]:
                heigth_point = point
            if right_point[0] < point[0]:
                right_point = point

        if left_point[1] <= right_point[1]:  # 正角度
            new_right_point = [right_point[0], heigth_point[1]]
            pts2 = np.float32([left_point, heigth_point, new_right_point])  # 字符只是高度需要改变
            pts1 = np.float32([left_point, heigth_point, right_point])
            M = cv2.getAffineTransform(pts1, pts2)
            dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
            point_limit(new_right_point)
            point_limit(heigth_point)
            point_limit(left_point)
            card_img = dst[int(left_point[1]):int(heigth_point[1]), int(left_point[0]):int(new_right_point[0])]
            if card_img.shape[:2][0] <= 10 or card_img.shape[:2][1] <= 10:
                print("hight or width too low")
                continue
            card_imgs.append(card_img)
            # cv2.imshow("card", card_img)
            # cv2.waitKey(0)
        elif left_point[1] > right_point[1]:  # 负角度

            new_left_point = [left_point[0], heigth_point[1]]
            pts2 = np.float32([new_left_point, heigth_point, right_point])  # 字符只是高度需要改变
            pts1 = np.float32([left_point, heigth_point, right_point])
            M = cv2.getAffineTransform(pts1, pts2)
            dst = cv2.warpAffine(oldimg, M, (pic_width, pic_hight))
            point_limit(right_point)
            point_limit(heigth_point)
            point_limit(new_left_point)
            card_img = dst[int(right_point[1]):int(heigth_point[1]), int(new_left_point[0]):int(right_point[0])]
            # card_img1 = oldimg[int(right_point[1]):int(heigth_point[1]), int(new_left_point[0]):int(right_point[0])]
            if card_img.shape[:2][0] <= 10 or card_img.shape[:2][1] <= 10:
                print("hight or width too low")
                continue
            card_imgs.append(card_img)
            # cv2.imshow("card_img1", card_img)
            # cv2.waitKey(0)

     # 开始使用颜色定位，排除不是车牌的矩形，目前只识别蓝、绿、黄车牌
    colors = []
    for card_index, card_img in enumerate(card_imgs):
        green = yello = blue = black = white = 0
        card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
        if card_img_hsv is None:
            continue
        row_num, col_num = card_img_hsv.shape[:2]
        # print("row_num:{}, col_num:{}".format(row_num, col_num))
        card_img_count = row_num * col_num
        # 确定车牌颜色
        for i in range(row_num):
            for j in range(col_num):
                H = card_img_hsv.item(i, j, 0)
                S = card_img_hsv.item(i, j, 1)
                V = card_img_hsv.item(i, j, 2)
                if 11 < H <= 34 and S > 34:
                    yello += 1
                elif 35 < H <= 99 and S > 34:
                    green += 1
                elif 99 < H <= 124 and S > 34:
                    blue += 1

                if 0 < H < 180 and 0 < S < 255 and 0 < V < 46:
                    black += 1
                elif 0 < H < 180 and 0 < S < 43 and 221 < V < 225:
                    white += 1
        color = "no"
        # 利用颜色点数来对轮廓进行排除
        limit1 = limit2 = 0
        if yello * 2 >= card_img_count:
            color = "yello"
            limit1 = 11
            limit2 = 34
        elif green * 2 >= card_img_count:
            color = "green"
            limit1 = 35
            limit2 = 99
        elif blue * 3 >= card_img_count:
            color = "blue"
            limit1 = 100
            limit2 = 124
        elif black + white >= card_img_count * 0.7:  # TODO
            color = "bw"
        print(color)
        colors.append(color)
        print(blue, green, yello, black, white, card_img_count)
        if limit1 == 0:
            continue
        if color == "yello":
            continue
        # cv2.imshow("color", card_img)
        # cv2.imshow("img", img)
        # cv2.waitKey(0)
        xl, xr, yh, yl = accurate_place(card_img_hsv, limit1, limit2, color)
        print("xl:{}, xr:{}, yh:{}, yl:{}".format(xl, xr, yh, yl))
        # print("row_num:{}, col_num:{}".format(row_num, col_num))
        if yl == yh and xl == xr:
            continue
        need_accurate = False
        if yl >= yh:
            yl = 0
            yh = row_num
            need_accurate = True
        if xl >= xr:
            xl = 0
            xr = col_num
            need_accurate = True
        if abs(yh-yl) < row_num*0.7:
            yl = 0
            yh = row_num
        card_imgs[card_index] = card_img[yl:yh, xl:xr] if color != "green" or yl < (yh - yl) // 4 else card_img[
                                                                                                       yl - (
                                                                                                            yh - yl) // 4:yh,
                                                                                                       xl:xr]
        print(xl, xr, yh, yl)

        if need_accurate:
            card_img = card_imgs[card_index]
            card_img_hsv = cv2.cvtColor(card_img, cv2.COLOR_BGR2HSV)
            xl, xr, yh, yl = accurate_place(card_img_hsv, limit1, limit2, color)
            if yl == yh and xl == xr:
                continue
            if yl >= yh:
                yl = 0
                yh = row_num
            if xl >= xr:
                xl = 0
                xr = col_num
        card_imgs[card_index] = card_img[yl:yh, xl:xr] if color != "green" or yl < (yh - yl) // 4 else card_img[
                                                                                                       yl - (
                                                                                                            yh - yl) // 4:yh,
                                                                                                    xl:xr]
        print(xl, xr, yh, yl)
        # print(len(card_imgs))
        if limit1 != 0:
            cv2.imshow("card_img", card_imgs[card_index])
            print(xl, xr, yh, yl)
            card1 = cv2.resize(card_imgs[card_index], (720, 180))
            cv2.imwrite("card_img_104_{}.jpg".format(card_index), card1)
            print(limit1)
            # cv2.imshow("oldimg{}".format(card_index), oldimg)
        else:
            print("未捕捉到车牌")
    cv2.waitKey(0)

# print('len(contours)', len(contours))
# cv2.imshow("kaiyunsuan", opening)
# cv2.imshow("yuantu", img)
# cv2.imshow("img_opening", img_opening)
# cv2.imshow("img_thresh", img_thresh)
# cv2.imshow("img_edge2", img_edge2)
# cv2.imshow("img", img)
# cv2.imshow("box", box)
# cv2.imwrite("img_edge2.jpg", img_edge2)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
if __name__ == '__main__':
    car_pic = "/home/python/Desktop/opencv_test/opencv_demo/pic/car21jinK88888.jpg"
    dingwei(car_pic)