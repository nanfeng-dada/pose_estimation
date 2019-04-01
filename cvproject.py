
import cv2
import numpy as np
import math
from matplotlib import pyplot as plt
# logitech camera
mtx = np.array(([728.97571788, 0.0, 310.08497498],
                [0.0, 730.5764702, 238.21487131],
                [0.0, 0.0, 1.0]), dtype=np.double)

distCoeffs = np.array(([-3.54744111e-02, 6.07436081e-01, 1.36986484e-03, -1.50088118e-03, -3.82286964e+00]),
                      dtype=np.double)
# my camera
# mtx = np.array(([553.61398782,  0.,             314.04195398],
#                 [  0.,          554.39959244,   237.482891],
#                 [0.0, 0.0, 1.0]), dtype=np.double)
# distCoeffs = np.array(([ 0.10298372, -0.34165059,  0.00298894, -0.00526068,  0.34475414]),
#                       dtype=np.double)


h_p = np.array([[[0, 0]], [[0, 210]], [[297, 210]], [[297, 0]]])
# A4
objp4 = np.array(([186, -105, 0], [186, 105, 0], [-111, 105, 0], [-111, -105, 0]), dtype=np.double)
# A3
objp8 = np.array(([148.32, -60, 0], [200.03, -25, 0], [158.03, -25, 0], [116.03, -25, 0],
                  [200.03, 25, 0], [158.03, 25, 0],[116.03, 25, 0] , [148.32, 60, 0]), dtype=np.double)
# A4
for i in objp8:
    for j in range(len(i)):
        # i[j]= i[j]*40.75/60.0
        # i[j] = i[j] * 10.10 / 14.832
        i[j] = i[j] * 136 / 200.03
# print(objp8)
objp12=np.vstack([objp4,objp8])

axis_3d=np.array(([0,0,0],[100, 0, 0],[0, 100, 0],[0, 0, 100]), dtype=np.double)
print(axis_3d)
str_axis=['O','X','Y','Z']

x = 0
y = 0
ctt = 0
no_ROI = 0
ell_all = []
aaa = np.array([])

# ---------------------------------------------------------
def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def eulerAnglesToRotationMatrix(theta):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])

    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])

    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))

    return R

# -----------------------------------------------------------

# cap1 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture('out1221.avi')
if (cap1.isOpened() == False):
    print("Error opening video stream or file")

while (cap1.isOpened()):
    ret, img = cap1.read()
    _, width, height = img.shape[::-1]
    my_edge = np.zeros((height, width), dtype=np.uint8)
    my_3d = np.ones((height, width,3), dtype=np.uint8) * 225

    # my_edge=cv2.cvtColor(my_edge,cv2.COLOR_BGR2GRAY)

    if ret == True:

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        gray = cv2.GaussianBlur(gray, (3, 3), 3)
        # ret, th1 = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)
        # th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        edge = cv2.Canny(gray, 186, 78)  # Canny边缘检测，参数可更改
        # cv2.imshow('canny', edge)
        # cv2.imshow('th1', th1)

        image, contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  #
        # image, contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  #
        if not (contours == []):
            c = max(contours, key=cv2.contourArea)
            # A4纸四个顶点
            point4 = cv2.approxPolyDP(c, 7, 1)
            c = max(contours, key=cv2.contourArea)
            Sc = cv2.contourArea(c)

            # print(Sc)

            if Sc > 3000:
                # cv2.drawContours(img, c, -1, (0, 0, 255), 2)
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(img, (x, y), (x + w, y + h), (153, 153, 0), 5)
                crop_edge = edge[y:y + h, x:x + w]
                my_edge[y:y + h, x:x + w] = edge[y:y + h, x:x + w]

                # cv2.imshow('later_edge', my_edge)
                image, contours, hierarchy = cv2.findContours(my_edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)  #
                # image, contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  #
                #     print('找到%d条边界' % len(contours))
                for cnt in contours:
                    # print('length=%d' % len(cnt))

                    if len(cnt) > 10 and len(cnt) < len(c) * 0.8:

                        S1 = cv2.contourArea(cnt)
                        ell = cv2.fitEllipse(cnt)
                        S2 = np.pi * ell[1][0] * ell[1][1]
                        if np.abs(S2) > 0.2 and (S1 / S2) > 0.2:  # 面积比例，可以更改，根据数据集。。。

                            if np.abs(x - ell[0][0]) + np.abs(y - ell[0][1]) > 2:
                                x_mean = ell[0][0]
                                y_mean = ell[0][1]
                                # print(str(S1) + "    " + str(S2)+"   "+str(x_mean)+"   "+str(y_mean))

                            else:
                                x_mean = np.sum(ell[0][0] + x) / 2
                                y_mean = np.sum(ell[0][1] + y) / 2
                                ell_all.append([x_mean, y_mean])

                            img = cv2.ellipse(img, ell, (0, 0, 255), 1)

                            x = ell[0][0]
                            y = ell[0][1]

                l = len(ell_all)
                # print('椭圆个数:%d' % l)
                # print(ell_all)

                print(l)
                # 椭圆编号
                # for i in range(l):
                #     x = int(ell_all[i][0])
                #     y = int(ell_all[i][1])
                #     cv2.putText(img, str(i), (x, y), 0, 1, (0, 255, 0),2)

                ctt = ctt + 1
                if ctt > 1 and l > 0:
                    ctt = 0
                    xc_sum=0
                    yc_sum=0
                    for i in range(l):
                        xc = int(ell_all[i][0])
                        yc = int(ell_all[i][1])
                        xc_sum=xc_sum+xc
                        yc_sum = yc_sum + yc
                        # cv2.putText(img, str(i), (xc, yc), 0, 0.4, (255, 0, 0), 1)
                    xm=xc_sum/l
                    ym = yc_sum / l

                    cv2.drawMarker(img, (int(xm), int(ym)), (255, 0, 0), 1,1,5)
                    dist = []
                    index = []
                    for i in range(len(point4)):
                        # cv2.drawMarker(img, (point4[i][0][0], point4[i][0][1]), (255, 0, 0), 2)
                        dist.append(np.abs(point4[i][0][0] - xm) + np.abs(point4[i][0][1] - ym))
                    dist_tmp = dist.copy()
                    dist_tmp.sort()
                    # print(dist_tmp)
                    for i in range(2):
                        index.append(dist.index(dist_tmp[i]))
                    # index[1]则为距离最小的点的索引
                    # print(index)
                    # 将A4纸四个角点按序排列
                    if index[0]-index[1]==1 or index[0]-index[1]==-3:
                        point4_regroup = np.array((point4[index[1] % 4], point4[(index[1] + 1) % 4],
                                                   point4[(index[1] + 2) % 4], point4[(index[1] + 3) % 4]),
                                                  dtype=np.double)
                    else:
                        point4_regroup = np.array((point4[index[0] % 4], point4[(index[0] + 1) % 4],
                                                   point4[(index[0] + 2) % 4], point4[(index[0] + 3) % 4]),
                                                  dtype=np.double)
                    # point4_regroup=np.array(point4_regroup)
                    # print(point4_regroup)
                    # for i in range(4):
                    #     cv2.putText(img, str(i), (int(point4_regroup[i][0][0]),int(point4_regroup[i][0][1])), 0, 1, (0, 255, 0), 2)
                    if l==8:# 若找到的是8个椭圆，则利用8+4个点计算位姿

                        M, _ = cv2.findHomography(point4_regroup, h_p, 0)
                        P3 = np.vstack([np.array(ell_all).T, np.ones((1, 8))])
                        Pc3 = np.dot(M, P3)
                        ell_weight = []
                        point8_regroup=[]
                        for i in range(l):
                            ell_weight.append(Pc3[0][i] + Pc3[1][i] * 2)

                        ell_weight_tmp = ell_weight.copy()
                        ell_weight_tmp.sort()
                        # print(ell_weight_tmp)
                        index = []

                        for i in range(8):
                            index.append(ell_weight.index(ell_weight_tmp[i]))

                        # print(index)
                        for i in range(8):
                            cv2.putText(img, str(i), (int(ell_all[index[i]][0]), int(ell_all[index[i]][1])), 0, 0.7,
                                        (255, 0, 0), 2)
                            point8_regroup.append([ell_all[index[i]]])

                        point8_regroup=np.array(point8_regroup)
                        print(point8_regroup)
                        point12=np.vstack([point4_regroup,point8_regroup])
                        found, rvec, tvec = cv2.solvePnP(objp12, point12, mtx, distCoeffs)

                    else:# 若找到的不是8个椭圆，则利用4个点计算位姿
                        found, rvec, tvec = cv2.solvePnP(objp4, point4_regroup, mtx, distCoeffs)
                    found, rvec, tvec = cv2.solvePnP(objp4, point4_regroup, mtx, distCoeffs)
                    print('----------')
                    # print('旋转向量')
                    # print(rvec)

                    theta = np.linalg.norm(rvec)
                    r = rvec / theta
                    R_ = np.array([[0, -r[2][0], r[1][0]],
                                   [r[2][0], 0, -r[0][0]],
                                   [-r[1][0], r[0][0], 0]])
                    R = np.cos(theta) * np.eye(3) + (1 - np.cos(theta)) * r * r.T + np.sin(theta) * R_
                    print('旋转矩阵')
                    print(R)

                    angle=rotationMatrixToEulerAngles(R)
                    print('欧拉角')
                    print(angle)
                    angle=angle*180/np.pi  +np.array(([0, 0, 0]),dtype=np.double)
                    angle=(angle*100)

                    print('平移向量')
                    print(tvec)
                    # print(np.sqrt((tvec[0]**2+tvec[1]**2+tvec[2]**2)))
                    Distance = int(np.linalg.norm(tvec))

                    print('距离=%f m' % (Distance / 1000))
                    cv2.putText(my_3d, 'D=' + str(Distance / 1000) + 'm', (10, 50), 0, 1, (255, 0, 0), 2)
                    cv2.putText(my_3d, 'X:' + str(int(angle[0] )/ 100) + 'deg', (10, 100), 0, 1, (255, 0, 0), 2)
                    cv2.putText(my_3d, 'Y:' + str(int(angle[1]) / 100) + 'deg', (10, 150), 0, 1, (255, 0, 0), 2)
                    cv2.putText(my_3d, 'Z:' + str(int(angle[2]) / 100) + 'deg', (10, 200), 0, 1, (255, 0, 0), 2)

                    axis_2d,_=cv2.projectPoints(axis_3d, rvec, tvec, mtx, distCoeffs)
                    print(axis_2d)
                    org_p=(int(axis_2d[0][0][0]+0.5),int(axis_2d[0][0][1]+0.5))
                    print(org_p)
                    for i in range(1,4):
                        ith_p = (int(axis_2d[i][0][0] + 0.5), int(axis_2d[i][0][1] + 0.5))
                        cv2.line(my_3d, org_p, ith_p, (0, 0, 255), 2, 16)
                        cv2.putText(my_3d,str_axis[i] , ith_p, 0, 1, (0, 255, 0), 2)
                    cv2.imshow('my_3d', my_3d)
                    print('*******************')

                ell_all = []
            else:

                print('Do not find ROI')
                # cv2.putText(img, 'Do not find ROI', (10, 50), 0, 1, (0, 255, 0), 2)

            # print(' ')
    else:
        print('play video ending')
        break

    cv2.imshow('res', img)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap1.release()
cv2.destroyAllWindows()