import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

# Harris和Shi-Tomasi角点检测算法，都具有旋转不变形，但是不具有尺度不变性
# 当小图可以检测到角点，但是当图片被放大后，使用同样的窗口就检测不到了

# 尺度不变特征转换SIFT，它用来侦测与描述影像中的局部性特征，它在空间尺度寻找极值点，并提取出其位置、尺度、旋转不变量。
# SIFT算法的实质是在不同的尺度空间上查找关键点(特征点)，并计算出关键点的方向。SIFT所查找到的关键点是一些十分突出，不会因光照，仿射变换和噪音等因素而变化的点，如角点、边缘点、暗区的亮点及亮区的暗点等

img = cv.imread('./image/tv.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# 2.sift关键点检测
# 2.1 实例化sift对象
sift = cv.xfeatures2d.SIFT_create()
# 2.2 关键点检测：kp关键点信息包括方向，尺度，位置信息，des是关键点的描述符
kp,des=sift.detectAndCompute(gray,None)
# 2.3 在图像上绘制关键点的检测结果
cv.drawKeypoints(img,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(img[:,:,::-1])
plt.show()

# 3.surf关键点检测
# 3.1 实例化surf对象
surf = cv.xfeatures2d.SURF_create(10000)

# 3.2 关键点检测：kp关键点信息包括方向，尺度，位置信息，des是关键点的描述符
kp,des=surf.detectAndCompute(gray,None)
# 3.3 在图像上绘制关键点的检测结果
cv.drawKeypoints(img,kp,img,flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
plt.imshow(img[:,:,::-1])
plt.show()

