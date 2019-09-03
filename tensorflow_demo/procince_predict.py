import os

import cv2
from PIL import Image
# import test_province
# import test_letters
# import test_digits
import time
import numpy as np
import tensorflow as tf
from pip._vendor.distlib._backport import shutil


SIZE = 1280
WIDTH = 32
HEIGHT = 40
# NUM_CLASSES = 7
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
    "zh_qing", "青",
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
nProvinceIndex = 0
time_begin = time.time()
# 定义输入节点，对应于图片像素值矩阵集合和图片标签(即所代表的数字)
x = tf.placeholder(tf.float32, shape=[None, SIZE])
y_ = tf.placeholder(tf.float32, shape=[None, 31])

x_image = tf.reshape(x, [-1, WIDTH, HEIGHT, 1])


# 定义卷积函数
def conv_layer(inputs, W, b, conv_strides, kernel_size, pool_strides, padding):
    L1_conv = tf.nn.conv2d(inputs, W, strides=conv_strides, padding=padding)
    L1_relu = tf.nn.relu(L1_conv + b)
    return tf.nn.max_pool(L1_relu, ksize=kernel_size, strides=pool_strides, padding='SAME')


# 定义全连接层函数
def full_connect(inputs, W, b):
    return tf.nn.relu(tf.matmul(inputs, W) + b)


# 加载训练模型
def province_test():
    saver_p = tf.train.import_meta_graph(
        "/home/python/Desktop/opencv_test/opencv_demo/tensorflow_demo/train-saver/province/model.ckpt.meta")
    with tf.Session() as sess_p:
        model_file = tf.train.latest_checkpoint("/home/python/Desktop/opencv_test/opencv_demo/tensorflow_demo/train-saver/province")
        saver_p.restore(sess_p, model_file)

        # 第一个卷积层
        W_conv1 = sess_p.graph.get_tensor_by_name("W_conv1:0")
        b_conv1 = sess_p.graph.get_tensor_by_name("b_conv1:0")
        conv_strides = [1, 1, 1, 1]
        kernel_size = [1, 2, 2, 1]
        pool_strides = [1, 2, 2, 1]
        L1_pool = conv_layer(x_image, W_conv1, b_conv1, conv_strides, kernel_size, pool_strides, padding='SAME')

        # 第二个卷积层
        W_conv2 = sess_p.graph.get_tensor_by_name("W_conv2:0")
        b_conv2 = sess_p.graph.get_tensor_by_name("b_conv2:0")
        conv_strides = [1, 1, 1, 1]
        kernel_size = [1, 1, 1, 1]
        pool_strides = [1, 1, 1, 1]
        L2_pool = conv_layer(L1_pool, W_conv2, b_conv2, conv_strides, kernel_size, pool_strides, padding='SAME')

        # 全连接层
        W_fc1 = sess_p.graph.get_tensor_by_name("W_fc1:0")
        b_fc1 = sess_p.graph.get_tensor_by_name("b_fc1:0")
        h_pool2_flat = tf.reshape(L2_pool, [-1, 16 * 20 * 32])
        h_fc1 = full_connect(h_pool2_flat, W_fc1, b_fc1)

        # dropout
        keep_prob = tf.placeholder(tf.float32)

        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        # readout层
        W_fc2 = sess_p.graph.get_tensor_by_name("W_fc2:0")
        b_fc2 = sess_p.graph.get_tensor_by_name("b_fc2:0")

        # 定义优化器和训练op
        conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
        list2 = []
        for root, dirs, files in os.walk("/home/python/Desktop/opencv_test/opencv_demo/tensorflow_demo/train_images/training-set/chinese-characters"):
            # 此处是最上面那个将车牌信息处理剪切成单个字符
            # 省份下标 1.temp  省份字母  2.tmp 根据上面剪切逻辑 依次类推
            rt = os.path.basename(root)
            if len(rt) > 3:
                continue
            right_val = PROVINCES[int(rt)]
            list1 =[]
        #     for file in files:
        #         path = os.path.join(root, file)
        #         img = Image.open(path)
        #         width = img.size[0]
        #         height = img.size[1]
        #         img_data = [[0] * SIZE for i in range(1)]
        #         for h in range(0, height):
        #             for w in range(0, width):
        #                 if img.getpixel((w, h)) < 190:
        #                     img_data[0][w + h * width] = 1
        #                 else:
        #                     img_data[0][w + h * width] = 0
        #         result = sess_p.run(conv, feed_dict={x: np.array(img_data), keep_prob: 1.0})
        #         # print("图片{}识别结果为:{}".format(file, PROVINCES[int(tf.argmax(result, 1).eval())]))
        #         if PROVINCES[int(tf.argmax(result, 1).eval())] != right_val:
        #             list1.append(file)
        #     # print(list1)
        #     print("{}识别率为:{:.2f}%".format(right_val, (len(files)-len(list1))/len(files)*100))
        #     list2.append((len(files)-len(list1))/len(files)*100)
        # print("-------------汉字总识别率为:{:.2f}%".format(sum(list2)/len(list2)))


        # for root, dirs, files in os.walk("/home/python/Desktop/charsChinese"):
        #     rt = os.path.basename(root)
        #     if not os.path.basename(root).startswith("zh_"):
        #         continue
        #     right_val = provinces[provinces.index(rt) + 1]
        #     # print(right_val)
        #     list1 =[]
        #     for file in files:
        #         path = os.path.join(root, file)
        #         img = Image.open(path)
        #         width = img.size[0]
        #         height = img.size[1]
        #         img_data = [[0] * SIZE for i in range(1)]
        #         for h in range(0, height):
        #             for w in range(0, width):
        #                 if img.getpixel((w, h)) < 190:
        #                     img_data[0][w + h * width] = 1
        #                 else:
        #                     img_data[0][w + h * width] = 0
        #         result = sess_p.run(conv, feed_dict={x: np.array(img_data), keep_prob: 1.0})
        #         predict_val = PROVINCES[int(tf.argmax(result, 1).eval())]
        #         # print("图片{}识别结果为:{}".format(file, predict_val))
        #         if predict_val != right_val:
        #             list1.append(file)
        #     # print(list1)
        #     print("{}识别率为:{:.2f}%".format(right_val, (len(files)-len(list1))/len(files)*100))
        #     list2.append((len(files)-len(list1))/len(files)*100)
        # print("-------------汉字总识别率为:{:.2f}%".format(sum(list2)/len(list2)))

        img = Image.open("/home/python/Desktop/opencv_test/opencv_demo/opencv_test1/chinese_test/bmp/car9zheF77777.bmp")
        width = img.size[0]
        height = img.size[1]
        img_data = [[0] * SIZE for i in range(1)]
        for h in range(0, height):
            for w in range(0, width):
                if img.getpixel((w, h)) < 190:
                    img_data[0][w + h * width] = 1
                else:
                    img_data[0][w + h * width] = 0
        result = sess_p.run(conv, feed_dict={x: np.array(img_data), keep_prob: 1.0})
        predict_val = PROVINCES[int(tf.argmax(result, 1).eval())]
        print(predict_val)






        #     max1 = 0
        #     max2 = 0
        #     max3 = 0
        #     max1_index = 0
        #     max2_index = 0
        #     max3_index = 0
        #     for j in range(7):
        #         if result[0][j] > max1:
        #             max1 = result[0][j]
        #             max1_index = j
        #             continue
        #         if (result[0][j] > max2) and (result[0][j] <= max1):
        #             max2 = result[0][j]
        #             max2_index = j
        #             continue
        #         if (result[0][j] > max3) and (result[0][j] <= max2):
        #             max3 = result[0][j]
        #             max3_index = j
        #             continue

            # nProvinceIndex = max1_index
            # print("概率：  [%s %0.2f%%]    [%s %0.2f%%]    [%s %0.2f%%]" % (
            #     PROVINCES[max1_index], max1 * 100, PROVINCES[max2_index], max2 * 100, PROVINCES[max3_index],
            #     max3 * 100))
        sess_p.close()

province_test()
