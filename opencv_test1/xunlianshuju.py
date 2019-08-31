import cv2
import os
import json
import numpy as np
from numpy.linalg import norm
SZ = 20
PROVINCE_START = 1000
from sklearn import metrics

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n * ang / (2 * np.pi))
        bin_cells = bin[:10, :10], bin[10:, :10], bin[:10, 10:], bin[10:, 10:]
        mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)


        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)


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


class StatModel(object):
    def load(self, fn):
        self.model = self.model.load(fn)

    def save(self, fn):
        self.model.save(fn)


class SVM(StatModel):
    def __init__(self, C=1, gamma=0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    # 训练svm
    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    # 字符识别
    def predict(self, samples):
        r = self.model.predict(samples)
        return r[1].ravel()


class CardPredictor:

    def __del__(self):
        self.save_traindata()

    def train_svm(self):
        # 识别英文字母和数字
        self.model = SVM(C=1, gamma=0.5)
        # 识别中文
        self.modelchinese = SVM(C=1, gamma=0.5)
        if os.path.exists("svm.dat"):
            self.model.load("svm.dat")
        else:
            chars_train = []
            chars_label = []

            for root, dirs, files in os.walk("/home/python/Desktop/opencv_test/opencv_test1/train/chars2"):
                if len(os.path.basename(root)) > 1:
                    continue
                root_int = ord(os.path.basename(root))
                for filename in files:
                    filepath = os.path.join(root, filename)
                    digit_img = cv2.imread(filepath)
                    digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
                    chars_train.append(digit_img)
                    # chars_label.append(1)
                    chars_label.append(root_int)

            chars_train = list(map(deskew, chars_train))
            chars_train = preprocess_hog(chars_train)
            # chars_train = chars_train.reshape(-1, 20, 20).astype(np.float32)
            chars_label = np.array(chars_label)
            print(chars_train.shape)
            self.model.train(chars_train, chars_label)
        if os.path.exists("svmchinese.dat"):
            self.modelchinese.load("svmchinese.dat")
        else:
            chars_train = []
            chars_label = []
            for root, dirs, files in os.walk("/home/python/Desktop/charsChinese"):
                if not os.path.basename(root).startswith("zh_"):
                    continue
                pinyin = os.path.basename(root)
                index = provinces.index(pinyin) + PROVINCE_START + 1  # 1是拼音对应的汉字
                for filename in files:
                    filepath = os.path.join(root, filename)
                    digit_img = cv2.imread(filepath)
                    digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
                    chars_train.append(digit_img)
                    # chars_label.append(1)
                    chars_label.append(index)
            chars_train = list(map(deskew, chars_train))
            chars_train = preprocess_hog(chars_train)
            # chars_train = chars_train.reshape(-1, 20, 20).astype(np.float32)
            chars_label = np.array(chars_label)
            print(chars_train.shape)
            self.modelchinese.train(chars_train, chars_label)

    def save_traindata(self):
        if not os.path.exists("svm.dat"):
            self.model.save("svm.dat")
        if not os.path.exists("svmchinese.dat"):
            self.modelchinese.save("svmchinese.dat")


if __name__ == '__main__':
    c = CardPredictor()
    c.train_svm()
    part_card = cv2.imread("/home/python/Desktop/opencv_test/opencv_test1/qiegezifu1_0.jpg", 0)
    old_pic = part_card
    w = abs(part_card.shape[1] - SZ) // 2
    part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    part_card = cv2.resize(part_card, (SZ, SZ), interpolation=cv2.INTER_AREA)
    part_card = deskew(part_card)
    part_card = preprocess_hog([part_card])
    resp = c.modelchinese.predict(part_card)  # 汉字
    charactor = provinces[int(resp[0]) - PROVINCE_START]    # 汉字
    # resp = c.model.predict(part_card)  # 英文数字
    # charactor = chr(resp[0])   # 英文数字
    print(charactor)

    # for root, dirs, files in os.walk("/home/python/Desktop/charsChinese/zh_xin"):
    #     if not os.path.basename(root).startswith("zh_"):
    #         continue
    #     pinyin = os.path.basename(root)
    #     index = provinces.index(pinyin) + PROVINCE_START + 1  # 1是拼音对应的汉字
    #     root2 = provinces[provinces.index(pinyin) + 1]
    #     count1 = 0
    #     list1 = []
    #     print("--------正确结果为:{}".format(root2))
    #     for j, filename in enumerate(files):
    #         filepath = os.path.join(root, filename)
    #         digit_img = cv2.imread(filepath)
    #         digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
    #         w = abs(digit_img.shape[1] - SZ) // 2
    #         part_card = cv2.copyMakeBorder(digit_img, 0, 0, w, w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    #         part_card = cv2.resize(part_card, (SZ, SZ), interpolation=cv2.INTER_AREA)
    #         part_card = deskew(part_card)
    #         part_card = preprocess_hog([part_card])
    #         resp = c.modelchinese.predict(part_card)
    #         charactor1 = provinces[int(resp[0]) - PROVINCE_START]
    #         print("图片{}识别结果为:{}".format(filename, charactor1))
    #         if root2 != charactor1:
    #             list1.append(filename)
    #     print(list1)
