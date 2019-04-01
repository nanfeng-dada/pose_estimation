#coding:utf-8
import cv2
import numpy as np
import glob


# 找棋盘格角点
# 阈值
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
#棋盘格模板规格
Chessboardsize=30     #mm
w = 5
h = 7
# 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
objp = np.zeros((w*h,3), np.float32)
objp[:,:2] = Chessboardsize*np.mgrid[0:w,0:h].T.reshape(-1,2)
print(objp)
print('---------------------')
# 交换xy坐标，非必要
# objp[:, [0, 1]] = objp[:, [1, 0]]
# print(objp)
# print(objp)

# 储存棋盘格角点的世界坐标和图像坐标对
objpoints = [] # 在世界坐标系中的三维点
imgpoints = [] # 在图像平面的二维点

#---------------获取图片---------------------------------------

# print('开始标定相机')
# cap = cv2.VideoCapture(1)
# pic_num=10
# item=0
# step=0
# while(1):
#     step=step+1
#     ret, frame = cap.read()
#
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     cv2.imshow("calib", gray)
#     ret, corners = cv2.findChessboardCorners(gray, (w, h), None,cv2.CALIB_CB_ADAPTIVE_THRESH)
#     cv2.waitKey(0)
#     if ret == True:
#         print('找到第%d张能检测角点的图' % (item+1))
#         keyvalue = cv2.waitKey(0)
#         if keyvalue & 0xFF == ord('o'):
#             item=item+1
#             cv2.imwrite("calib%d.jpg" % item, frame)
#             if item>=pic_num:
#                 break
#             else:
#                 continue
#         # elif keyvalue & 0xFF == ord('r'):
#         else:
#             continue
#     else:
#         print('step：%d，未找到角点' % step)
# cap.release()
# cv2.destroyWindow('calib')

#------------------------------------------------------
images = glob.glob('calib_img\calib*.jpg')
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # print(img.shape)
    # print(img.shape[::-1])
    # 找到棋盘格角点
    cv2.imshow("pic", gray)

    cv2.waitKey(0)
    ret, corners = cv2.findChessboardCorners(gray, (w,h),None,cv2.CALIB_CB_ADAPTIVE_THRESH)
    # 如果找到足够点对，将其存储起来
    if ret == True:
        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)
        print(corners)
        print('+++++++++++++++++++++++++++++')
        objpoints.append(objp)


        cv2.drawChessboardCorners(img, (w,h), corners, ret)
        cv2.imshow('findCorners',img)
        # cv2.imwrite('corner2.jpg',img)


        cv2.waitKey(0)

# print(objpoints)

cv2.destroyAllWindows()

# opencv自带标定算法
# 标定
# w,h=gray.shape[::-1]
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#
# cv2.Rodrigues()
# cv2.rotate()
print(ret)
print('---------')

print(mtx)
print('---------')
print(dist)
print('---------')

print(rvecs)
print('---------')
print(tvecs)





