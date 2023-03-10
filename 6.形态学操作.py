import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

# 腐蚀和膨胀都是针对白色部分（高亮部分）而言的
# 膨胀就是使高亮部分扩张，效果图拥有比原图更大的高亮区域       腐蚀是原图中的高亮区域被蚕食，效果图拥有比原图更小的高亮区域。
# 膨胀是求局部最大值的操作。（高亮变大）  作用是将与物体接触的所有背景点合并到物体中，使目标增大，可添补目标中的孔洞。  --- 或操作
# 腐蚀是求局部最小值的操作。（高亮变小）  作用是消除物体边界点，使目标缩小，可以消除小于结构元素的噪声点。           --- 与操作
img = cv.imread('data/letter.jpg')
plt.imshow(img[:,:,::-1])
plt.show()
# 创建核结构
kernel = np.ones((5,5),np.uint8)

# 腐蚀
img1 = cv.erode(img,kernel)
plt.imshow(img1[:,:,::-1])
plt.show()

# 膨胀
img2 = cv.dilate(img,kernel)
plt.imshow(img2[:,:,::-1])
plt.show()

# 开运算是先腐蚀后膨胀，其作用是：分离物体，消除小区域。特点：消除噪点，去除小的干扰块，而不影响原来的图像。
open = cv.imread('data/open_letter.jpg')
plt.imshow(open[:,:,::-1])
plt.show()

kernel1 = np.ones((10,10),np.uint8)
cvopen = cv.morphologyEx(open,cv.MORPH_OPEN,kernel1)
plt.imshow(cvopen[:,:,::-1])
plt.show()

# 闭运算与开运算相反，是先膨胀后腐蚀，作用是消除/“闭合”物体里面的孔洞，特点：可以填充闭合区域。
close = cv.imread('data/close_letter.jpg')
plt.imshow(close[:,:,::-1])
plt.show()

cvclose = cv.morphologyEx(open,cv.MORPH_CLOSE,kernel1)
plt.imshow(cvclose[:,:,::-1])
plt.show()

# 礼帽运算用来分离比邻近点亮一些的斑块。当一幅图像具有大幅的背景的时候，而微小物品比较有规律的情况下，可以使用顶帽运算进行背景提取。 原图像与开运算之差
# dst = src - open(src,element)
top = cv.morphologyEx(open,cv.MORPH_OPEN,kernel1)
plt.imshow(top[:,:,::-1])
plt.show()


# 黑帽运算后的效果图突出了比原图轮廓周围的区域更暗的区域，且这一操作和选择的核的大小相关。 闭运算与原图像之差
# dst = close(src,element) - src
black = cv.morphologyEx(close,cv.MORPH_CLOSE,kernel1)
plt.imshow(black[:,:,::-1])
plt.show()