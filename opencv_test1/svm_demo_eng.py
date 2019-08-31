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


# 获得数据的特征向量
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

    # 识别英文字母和数字
    model = SVM(C=1, gamma=0.5)
    if os.path.exists("svm.dat"):
        model.load("svm.dat")  # 如果存在，不再训练，直接导入训练好的结果
    else:
        chars_train = []
        chars_label = []

        for root, dirs, files in os.walk("/home/python/Desktop/opencv_test/opencv_test1/train/chars2"):

            """
            root:所指的是当前正在遍历的这个文件夹的本身的地址
            dirs:是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
            files:同样是 list , 内容是该文件夹中所有的文件(不包括子目录)

            目录下保存有数字和大写字母图片，用于训练
            """
            # os.path.basename(),返回path最后的文件名
            # 例：root=train\chars2\7,那么os.path.basename(root)=7

            if len(os.path.basename(root)) > 1:  # 目录是单个字母或者数字
                continue
            root_int = ord(os.path.basename(root))  # 转化为ASCII字符
            for filename in files:
                filepath = os.path.join(root, filename)
                digit_img = cv2.imread(filepath)

                # https://www.aiuai.cn/aifarm365.html
                # 把图片转化为灰度图
                # print 'filename: '+filename
                digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)

                # print digit_img.shape #打印测试一下，可以看到是单通道的灰度图。(20L, 20L)
                # 采用PIL库可视化显示一下灰度图
                # img_pil = Image.fromarray(digit_img); #Image.fromarray实现array到image的转换
                # img_pil.show()

                # print '***************************************\n'
                # print digit_img  #打印一下转化的灰度图矩阵

                chars_train.append(digit_img)  # 训练样本集合
                # chars_label.append(1)
                chars_label.append(root_int)  # 训练样本标签，这里用字符的ASCII表示

        # end_time = datetime.datetime.now()
        # print('----------')
        # print(start_time, end_time, (end_time - start_time).seconds)

        # map() 会根据提供的函数对指定序列做映射。
        # 第一个参数 function 以参数序列中的每一个元素调用 function 函数，返回包含每次 function 函数返回值的新列表。
        # 把灰度图的训练样本集合中每个元素逐一送入deskew函数进行抗扭斜处理--也就是把图片摆正
        chars_train = list(map(deskew, chars_train))

        chars_train = hog(chars_train)  # 获得特征向量

        # print '---chars_label---'
        # print chars_label

        # chars_train = chars_train.reshape(-1, 20, 20).astype(np.float32)
        chars_label = np.array(chars_label)

        model.train(chars_train, chars_label)  # SVM训练，opencv自带

    if not os.path.exists("svm.dat"):
        model.save("svm.dat")

    # 测试一下训练结果
    img = cv2.imread('/home/python/Desktop/opencv_test/opencv_test1/qiegezifu3.jpg')
    img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (SZ, SZ), interpolation=cv2.INTER_AREA)
    # print img
    resp = model.predict(hog([img]))
    charactor = chr(resp[0])
    print('--result---' + charactor)


train_svm()

