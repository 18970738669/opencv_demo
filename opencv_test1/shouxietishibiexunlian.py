import cv2
import os
import json
import numpy as np
from numpy.linalg import norm
SZ = 20


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
        self.modelshouxie = SVM(C=1, gamma=0.5)
        if os.path.exists("svmshouxie.dat"):
            self.modelshouxie.load("svmshouxie.dat")
        else:
            chars_train = []
            chars_label = []

            for root, dirs, files in os.walk("/home/python/Desktop/opencv_test/opencv_test1/train_shouxie/Validation"):
                if len(os.path.basename(root)) > 1:
                    continue
                root_int = ord(os.path.basename(root))
                print("-" * 20)
                print(os.path.basename(root))
                for filename in files:
                    if filename == ".DS_Store":
                        continue
                    print(filename)
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
            self.modelshouxie.train(chars_train, chars_label)

    def save_traindata(self):
        if not os.path.exists("svmshouxie.dat"):
            self.modelshouxie.save("svmshouxie.dat")


if __name__ == '__main__':
    c = CardPredictor()
    c.train_svm()
    part_card = cv2.imread("/home/python/Desktop/opencv_test/opencv_test1/qiegezifu1_3.jpg")
    w = abs(part_card.shape[1] - SZ) // 2
    part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    part_card = cv2.resize(part_card, (SZ, SZ), interpolation=cv2.INTER_AREA)
    part_card = preprocess_hog([part_card])
    resp = c.modelshouxie.predict(part_card)
    print(chr(resp[0]))
    # img = cv2.imread("/home/python/Desktop/opencv_test/samoye1.jpg")
    # cv2.imshow("img", img)
    # cv2.waitKey(0)