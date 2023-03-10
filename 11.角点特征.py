import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# 角点是图像很重要的特征,对图像图形的理解和分析有很重要的作用。角点在三维场景重建运动估计，目标跟踪、目标识别、图像配准与匹配等计算机视觉领域起着非常重要的作用。
# 在现实世界中，角点对应于物体的拐角，道路的十字路口、丁字路口等

# 图像特征要有区分性，容易被比较。一般认为角点，斑点等是较好的图像特征
# 特征检测：找到图像中的特征
# 特征描述：对特征及其周围的区域进行描述

# E(u,v)反应窗口移动后亮度的变化，w(x,y)表示权重，高斯--距离中心的占比大  均值--一样大   E = w(x,y)[移动后-移动前]^2

# Harris角点检测  通过图像的局部的小窗口观察图像，角点的特征是窗口沿任意方向移动都会导致图像灰度的明显变化。
img = cv.imread('./image/chessboard.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray = np.float32(gray)
plt.imshow(gray,cmap=plt.cm.gray)
plt.show()

# 返回每个点的R值
dst = cv.cornerHarris(gray,2,3,0.04)
print(dst)
# 设置阈值，将角点绘制出来，阈值根据图像进行选择
img[dst>0.001*dst.max()] = [0,0,255]  # dst>0.001 * dst.max() 被认为是交点，用红色的标出来
plt.imshow(img[:,:,::-1])
plt.show()

print('---------next program---------')

# shi-tomas角点检测 看出来只有当 λ1 和 λ 2 都大于最小值时，才被认为是角点。
# 搜索到的角点，在这里所有低于质量水平的角点被排除掉，然后把合格的角点按质量排序，然后将质量较好的角点附近（小于最小欧式距离）的角点删掉，最后找到maxCorners个角点返回
img = cv.imread('./image/tv.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
plt.imshow(gray,cmap=plt.cm.gray)
plt.show()

coners = cv.goodFeaturesToTrack(gray,1000,0.01,10)
print(coners)

for i in coners:
    x,y = i.ravel()  # .ravel() 将数组拉成一维数组，平铺开
    cv.circle(img,(int(x),int(y)),2,(0,0,255),-1)

plt.imshow(img[:,:,::-1])
plt.show()


