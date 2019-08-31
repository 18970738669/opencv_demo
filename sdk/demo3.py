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

image = get_file_content('/home/python/Desktop/opencv_test/sdk/fapiao1.png')

""" 调用增值税发票识别 """
print(client.vatInvoice(image))
print("购买方名称:{}\n纳税人识别号:{}\n")

"""
{
  "log_id": "5425496231209218858",
  "words_result_num": 29,
  "words_result": {
    "InvoiceNum": "14641426",
    "SellerName": "上海易火广告传媒有限公司",
    "CommodityTaxRate": [
      {
        "word": "6%",
        "row": "1"
      }
    ],
    "SellerBank": "中国银行南翔支行446863841354",
    "Checker": ":沈园园",
    "TotalAmount": "94339.62",
    "CommodityAmount": [
      {
        "word": "94339.62",
        "row": "1"
      }
    ],
    "InvoiceDate": "2016年06月02日",
    "CommodityTax": [
      {
        "word": "5660.38",
        "row": "1"
      }
    ],
    "PurchaserName": "百度时代网络技术(北京)有限公司",
    "CommodityNum": [
      {
        "word": "",
        "row": "1"
      }
    ],
    "PurchaserBank": "招商银行北京分行大屯路支行8661820285100030",
    "Remarks": "告传",
    "Password": "074/45781873408>/6>8>65*887676033/51+<5415>9/32--852>1+29<65>641-5>66<500>87/*-34<943359034>716905113*4242>",
    "SellerAddress": ":嘉定区胜辛南路500号15幢1161室55033753",
    "PurchaserAddress": "北京市海淀区东北旺西路8号中关村软件园17号楼二属A2010-59108001",
    "InvoiceCode": "3100153130",
    "CommodityUnit": [
      {
        "word": "",
        "row": "1"
      }
    ],
    "Payee": ":徐蓉",
    "PurchaserRegisterNum": "110108787751579",
    "CommodityPrice": [
      {
        "word": "",
        "row": "1"
      }
    ],
    "NoteDrawer": "沈园园",
    "AmountInWords": "壹拾万圆整",
    "AmountInFiguers": "100000.00",
    "TotalTax": "5660.38",
    "InvoiceType": "专用发票",
    "SellerRegisterNum": "913101140659591751",
    "CommodityName": [
      {
        "word": "信息服务费",
        "row": "1"
      }
    ],
    "CommodityType": [
      {
        "word": "",
        "row": "1"
      }
    ]
  }
}"""