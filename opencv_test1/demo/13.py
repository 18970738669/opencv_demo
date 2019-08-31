import cv2

img = cv2.imread("/home/python/Desktop/opencv_test/opencv_test1/card_img_100.jpg")

img = cv2.resize(img, (600, 280), interpolation=cv2.INTER_AREA)
cv2.imshow("img", img)
cv2.waitKey(0)