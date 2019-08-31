import datetime

import cv2
import numpy as np
from numpy.linalg import norm
import os

SZ = 20  # 训练图片长宽

def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

class StatModel(object):
    def load(self, fn):
        self.model = self.model.load(fn)

    def save(self, fn):
        self.model.save(fn)


# 利用OpenCV中的SVM进行机器学习
class SVM(StatModel):
    def __init__(self, C=1, gamma=0.5):
        self.model = cv2.ml.SVM_create()  # 创建SVM model

        # 属性设置
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)  # 径向基核函数（(Radial Basis Function），比较好的选择，gamma>0；
        self.model.setType(cv2.ml.SVM_C_SVC)

    # 训练svm
    def train(self, samples, responses):  # SVM的训练函数
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    # 字符识别
    def predict(self, samples):
        r = self.model.predict(samples)
        return r[1].ravel()
        # 来自opencv的sample，用于svm训练


def hog(digits):
    samples = []

    '''
    step1.先计算图像 X 方向和 Y 方向的 Sobel 导数。
    step2.然后计算得到每个像素的梯度角度angle和梯度大小magnitude。
    step3.把这个梯度的角度转换成 0至16 之间的整数。
    step4.将图像分为 4 个小的方块，对每一个小方块计算它们梯度角度的直方图（16 个 bin），使用梯度的大小做权重。
    这样每一个小方块都会得到一个含有 16 个值的向量。
    4 个小方块的 4 个向量就组成了这个图像的特征向量（包含 64 个值）。
    这就是我们要训练数据的特征向量。
    '''

    for img in digits:
        # plt.subplot(221)
        # plt.imshow(img,'gray')
        # step1.计算图像的 X 方向和 Y 方向的 Sobel 导数
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)  # step2.笛卡尔坐标（直角/斜角坐标）转换为极坐标, → magnitude, angle

        bin_n = 16
        bin = np.int32(bin_n * ang / (2 * np.pi))  # step3. quantizing binvalues in (0...16)。2π就是360度。

        # step4. Divide to 4 sub-squares
        bin_cells = bin[:10, :10], bin[10:, :10], bin[:10, 10:], bin[10:, 10:]
        mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]

        # zip() 函数用于将可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表。
        # a = [1,2,3];b = [4,5,6];zipped = zip(a,b)  结果[(1, 4), (2, 5), (3, 6)]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)  # hist is a 64 bit vector
        # plt.subplot(223)
        # plt.plot(hist)

        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps
        # plt.subplot(224)
        # plt.plot(hist)
        # plt.show()

        samples.append(hist)

    return np.float32(samples)

def train_svm():

    chinesemodel = SVM(C=1, gamma=0.5)
    if os.path.exists("svmchi.dat"):
        chinesemodel.load("svmchi.dat")
    else:
        chars_train = []
        chars_label = []


