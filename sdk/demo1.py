import os

from aip import AipOcr

""" 你的 APPID AK SK """
APP_ID = '17071937'
API_KEY = 'h69LielxQzwCOW0YYilVIFUv'
SECRET_KEY = 'pVefAGr8WveDV5wvqFW8S90g3Kr9eh2L'

client = AipOcr(APP_ID, API_KEY, SECRET_KEY)


""" 读取图片 """
def get_file_content(filePath):
    with open(filePath, 'rb') as fp:
        return fp.read()

image = get_file_content('/home/python/Desktop/opencv_test/opencv_test1/test_pic/car15.jpg')

""" 调用车牌识别 """
print(client.licensePlate(image)["words_result"]["number"])

# """ 如果有可选参数 """
# options = {}
# options["multi_detect"] = "true"
#
# """ 带参数调用车牌识别 """
# client.licensePlate(image, options)
# for root, dirs, files in os.walk("/home/python/Desktop/opencv_test/pic"):
#     count = 0
#     # unshibie = []
#     for i, file in enumerate(files):
#         car_pic = os.path.join(root, file)
#         image = get_file_content(car_pic)
#         print(car_pic)
#         c = client.licensePlate(image)["words_result"]["number"]
#         print(c)
#     break