import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# 图像缩放
img = cv.imread('data/tang.jpg')
plt.imshow(img[:,:,::-1])
plt.show()
rows, cols = img.shape[:2]   # 获取时，先行数、列数

# 绝对坐标
res = cv.resize(img,(2*cols,2*rows))   # 缩放时，输出大小 (2*cols,2*rows) 先列后行
plt.imshow(res[:,:,::-1])
plt.show()

# 相对坐标
res1 = cv.resize(img,None,fx=0.5,fy=0.5)
plt.imshow(res1[:,:,::-1])
plt.show()

# 图像平移  原点在左上角，[x y 1] * [[1 0 p],[0 1 q] ...]
M = np.float32([[1,0,100],[0,1,50]])   # 移动矩阵
res2 = cv.warpAffine(img,M,(cols,rows))   # 移动后生成图像的大小  (cols,rows) 先列后行
plt.imshow(res2[:,:,::-1])
plt.show()

M = np.float32([[1,0,100],[0,1,50]])
res2 = cv.warpAffine(img,M,(2*cols,2*rows))   # 移动了之后图像的大小 (cols,rows) 先列后行
plt.imshow(res2[:,:,::-1])
plt.show()  # 但是图像本身没有变大，在图像平移的基础上是黑色区域变大了，导致图片增大

# 图像旋转<点的位置不动，坐标轴动了>   x'=rcos(x-p)
# 原图像中的坐标原点在图像的左上角(0,0,)，经过旋转后图像的大小会有所变化，原点需要修正。 方便理解 ==> 平移原点 ==> 旋转图像
# 假设使用2x2的矩阵，是没有办法描述平移操作的，只有引入3x3矩阵形式，才能统一描述二维中的平移、旋转、缩放操作

M = cv.getRotationMatrix2D((cols/2,rows/2),45,0.5)  # 指定旋转中心<先列后行>，指定角度，缩放大小，生成旋转矩阵
dst = cv.warpAffine(img,M,(cols,rows))  # 生成图片大小 先列后行
plt.imshow(dst[:,:,::-1])
plt.show()

# 仿射变换
# 对图像进行缩放，旋转，翻转和平移等操作的组合，制定一个2*3的矩阵
# 从原始图像中找到三个点以及他们在输出图像中的位置，创建一个变换矩阵
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[100,100],[200,50],[100,250]])
# 构建变换矩阵
M = cv.getAffineTransform(pts1,pts2)
res4 = cv.warpAffine(img,M,(cols,rows))
plt.imshow(res4[:,:,::-1])
plt.show()

# 透射变换
# 利用透视中心、像点、目标点三点共线的条件，按透视旋转定律使承影面（透视面）绕迹线（透视轴）旋转某一角度，破坏原有的投影光线束，仍能保持承影面上投影基核图形不变的变换
pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[100,145],[300,100],[80,290],[310,300]])
T = cv.getPerspectiveTransform(pts1,pts2)
res5 = cv.warpPerspective(img,T,(cols,rows))
plt.imshow(res5[:,:,::-1])
plt.show()

# 图像金字塔,上采样分辨率增高，下采样分辨率下降
up_img = cv.pyrUp(img)
down_img = cv.pyrUp(img)
plt.imshow(up_img[:,:,::-1])
plt.show()
plt.imshow(down_img[:,:,::-1])
plt.show()

