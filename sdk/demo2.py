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

image = get_file_content('/home/python/Desktop/opencv_test/opencv_test1/card_img_104_0.jpg')

""" 调用通用文字识别（高精度版） """
print(client.basicAccurate(image))

# """ 如果有可选参数 """
# options = {}
# options["detect_direction"] = "true"
# options["probability"] = "true"
#
# """ 带参数调用通用文字识别（高精度版） """
# client.basicAccurate(image, options)