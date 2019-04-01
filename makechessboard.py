""""""
"""
生成棋盘格标定板
"""
import cv2
import numpy as np
# times：放大倍数
#width，height,length单位：pixel
times=2
width,height,length = [x*times for x in (450,350,50)]


image = np.zeros((width,height),dtype = np.uint8)
print(image.shape[0],image.shape[1])

for j in range(height):
    for i in range(width):
        if((int)(i/length) + (int)(j/length))%2:
            image[i,j] = 255;
cv2.imwrite("chess.jpg",image)
cv2.imshow("chess",image)
cv2.waitKey(0)
