import cv2
import os
from PIL import Image
provinces = [
    "zh_cuan", "川",
    "zh_e", "鄂",
    "zh_gan", "赣",
    "zh_gan1", "甘",
    "zh_gui", "贵",
    "zh_gui1", "桂",
    "zh_hei", "黑",
    "zh_hu", "沪",
    "zh_ji", "冀",
    "zh_jin", "津",
    "zh_jing", "京",
    "zh_jl", "吉",
    "zh_liao", "辽",
    "zh_lu", "鲁",
    "zh_meng", "蒙",
    "zh_min", "闽",
    "zh_ning", "宁",
    "zh_qing", "靑",
    "zh_qiong", "琼",
    "zh_shan", "陕",
    "zh_su", "苏",
    "zh_sx", "晋",
    "zh_wan", "皖",
    "zh_xiang", "湘",
    "zh_xin", "新",
    "zh_yu", "豫",
    "zh_yu1", "渝",
    "zh_yue", "粤",
    "zh_yun", "云",
    "zh_zang", "藏",
    "zh_zhe", "浙"
]
PROVINCES = ("京", "闽", "粤", "苏", "沪", "浙", "川", "鄂", "赣", "甘", "贵", "桂", "黑", "冀", "津", "吉", "辽", "鲁", "蒙"
             , "宁", "青", "琼", "陕", "晋", "皖", "湘", "新", "豫", "渝", "云", "藏")
# 图像放大原来的两倍
# img = cv2.imread("/home/python/Desktop/opencv_test/opencv_test1/train_shouxie/Validation/#/__0_121440.png", 0)
# height, width = img.shape[:2]
# res = cv2.resize(img, (2 * width, 2 * height), interpolation=cv2.INTER_CUBIC)
# cv2.imshow("res", res)
# gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("img", img)
# cv2.imshow("gray_img", gray_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# for root, dirs, files in os.walk("/home/python/Desktop/opencv_test/opencv_test1/train_shouxie/Validation"):
#     print("root:{}\ndirs:{}\nfiles:{}".format(root, dirs, files))
path = "/home/python/Desktop/opencv_test/opencv_test1/chinese_test/qiegezifulu_0.jpg"
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.resize(img, (32, 40), interpolation=cv2.INTER_AREA)
a = path.split(".")
new_path = a[0] + ".bmp"
cv2.imwrite(path, img)
image = Image.open(path)
new_pic = image.convert("L")
new_pic.save(new_path)
os.remove(path)
print("图片{}修改格式成功".format(os.path.basename(path)))
# cv2.imshow("img", img)
# # os.remove("/home/python/Desktop/opencv_test/opencv_test1/test_pic/car1.jpg")
# cv2.waitKey(0)


# for root, dirs, files in os.walk("/home/python/Desktop/charsChinese1"):
#     if not os.path.basename(root).startswith("zh_"):
#         continue
#     if os.path.basename(root) in ["chinese-characters", "zh_jing", "zh_min", "zh_yue", "zh_su", "zh_hu", "zh_zhe"]:
#         continue
#     for j, filename in enumerate(files):
#         filepath = os.path.join(root, filename)
#         digit_img = Image.open(filepath)
#         new_pic = digit_img.convert("L")
#         a = filename.split(".")
#         new_filename = a[0] + ".bmp"
#         new_filepath = os.path.join(root, new_filename)
#         new_pic.save(new_filepath)
#         os.remove(filepath)
#         digit_img = cv2.imread(filepath)
#         img = cv2.resize(digit_img, (32, 40), interpolation=cv2.INTER_AREA)
#         os.remove(filepath)
#         cv2.imwrite(filepath, img)
#         print("改写完一张{}".format(filename))


# img = cv2.imread("/home/python/Desktop/opencv_test/opencv_test1/train/charsChinese/zh_min/1140-0-9.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img = cv2.resize(img, (32, 40), interpolation=cv2.INTER_AREA)
# cv2.imwrite("/home/python/Desktop/opencv_test/opencv_test1/qiegezifu1_.jpg", img)

