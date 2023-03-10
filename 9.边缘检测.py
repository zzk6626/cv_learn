import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# 了解Sobel算子，Scharr算子和拉普拉斯算子  /   掌握canny边缘检测的原理及应用
# 基于搜索：通过寻找图像一阶导数中的最大值来检测边界，然后利用计算结果估计边缘的局部方向，通常采用梯度的方向，并利用此方向找到局部梯度模的最大值，代表算法是Sobel算子和Scharr算子
# 基于零穿越：通过寻找图像二阶导数零穿越来寻找边界，代表算法是Laplacian算子

# Sobel边缘检测算法比较简单，实际应用中效率比canny边缘检测效率要高，但是边缘不如Canny检测的准确，但是很多实际应用的场合，sobel边缘却是首选，Sobel算子是高斯平滑与微分操作的结合体
# 所以其抗噪声能力很强，用途较多。尤其是效率要求较高，而对细纹理不太关心的时候。
img = cv.imread('./image/horse.jpg',0)
plt.imshow(img,cmap=plt.cm.gray)
plt.show()

# 分别计算x，y方向的Sobel算子Sobel函数求完导数后会有负值，还有会大于255的值。而原图像是uint8，即8位无符号数，所以Sobel建立的图像位数不够，会有截断。
# 因此要使用16位有符号的数据类型，即cv2.CV_16S。处理完图像后，再使用cv2.convertScaleAbs()函数将其转回原来的uint8格式，否则图像无法显示。
x = cv.Sobel(img,cv.CV_16S,1,0)
y = cv.Sobel(img,cv.CV_16S,0,1)
absx = cv.convertScaleAbs(x)
absy = cv.convertScaleAbs(y)
res1 = cv.addWeighted(absx,0.5,absy,0.5,0)  # 0.5x' + 0.5y' + 0
plt.imshow(res1,cmap=plt.cm.gray)
plt.show()

# Schaar算子，跟Sobel算子差别不大，修改了权重[3 10 3]
x = cv.Sobel(img,cv.CV_16S,1,0,ksize=-1)
y = cv.Sobel(img,cv.CV_16S,0,1,ksize=-1)
absx = cv.convertScaleAbs(x)
absy = cv.convertScaleAbs(y)
res2 = cv.addWeighted(absx,0.5,absy,0.5,0)  # 0.5x' + 0.5y' + 0
plt.imshow(res2,cmap=plt.cm.gray)
plt.show()

# Laplacian算子，Laplacian是利用二阶导数来检测边缘
res3 = cv.Laplacian(img,cv.CV_16S)
res3 = cv.convertScaleAbs(res3)
plt.imshow(res3,cmap=plt.cm.gray)
plt.show()

# Canny边缘检测
# Canny边缘检测算法是一种非常流行的边缘检测算法
# 1.噪声去除   5*5高斯滤波器
# 2.计算梯度   找到梯度的大小和方向 如果某个像素点是边缘，则其梯度方向总是垂直与边缘垂直。梯度方向被归为四类：垂直，水平，和两个对角线方向。二维图像
# 3.非极大值抑制 对整幅图像进行扫描，去除那些非边界上的点。对每一个像素进行检查，看这个点的梯度是不是周围具有相同梯度方向的点中最大的。
# 4.滞后阈值 现在要确定真正的边界。 我们设置两个阈值： minVal 和 maxVal。 当图像的灰度梯度高于 maxVal 时被认为是真的边界， 低于 minVal 的边界会被抛弃。介于中间，判断相连性

Gsr = cv.GaussianBlur(img, (5, 5), 0)
res4 = cv.Canny(Gsr,0,100)
plt.imshow(res4,cmap=plt.cm.gray)
plt.show()
