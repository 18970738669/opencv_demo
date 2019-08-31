import cv2
import numpy as np


cnt = np.array([[11,22],[33,44],[55,66],[77,88]])
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
print(rect[0])
print(rect[1])
print(rect[2])
cv2.imshow("box", box)
cv2.waitKey(0)
cv2.destroyAllWindows()