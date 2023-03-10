import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# 获取并修改像素点
# 通过行和列的坐标值获取该像素点的像素值，对于BGR图像，它返回一个蓝，绿，红值得数组。对于灰度图像，仅返回相应的强度值。
img = np.zeros((256,256,3),np.uint8)
plt.imshow(img[:,:,::-1])
plt.show()
print(img[100,100])
print(img[100,100,0])  # 蓝色通道的值

img[100,100] = (0,0,255)
plt.imshow(img[:,:,::-1])
plt.show()

# 获取图像属性  行列数/图像大小/像素点
print(img.shape)
print(img.dtype)
print(img.size)  # 像素点

# 图像通道的拆分与合并
# 通道拆分
# b,g,r = cv.split(img)
# 通道合并
# img = cv.merge((b,g,r))

img1 = cv.imread('data/1.jpg')
plt.imshow(img1[:,:,::-1])
plt.show()
b,g,r = cv.split(img1)
plt.imshow(b,cmap=plt.cm.gray)
plt.show()
img2 = cv.merge((b,g,r))
plt.imshow(img2[:,:,::-1])
plt.show()

# 色彩空间的改变  BGR -> Gray  BGR -> HSV
gray = cv.cvtColor(img1,cv.COLOR_BGR2GRAY)
plt.imshow(gray,cmap=plt.cm.gray)
plt.show()

hsv = cv.cvtColor(img1,cv.COLOR_BGR2HSV)
plt.imshow(hsv)
plt.show()