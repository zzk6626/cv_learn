import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

img = cv.imread('image/cat.jpeg',0)
plt.imshow(img,cmap=plt.cm.gray)
plt.show()

# 绘制直方图,满足灰度值限制的像素点个数
hist = cv.calcHist([img],[0],None,[256],[0,256])
plt.figure()
plt.plot(hist)
plt.show()

# 掩膜的应用，通常使用二维矩阵数组进行掩膜，1值的区域而被处理，0值区域被屏蔽，不会处理
# 提取感兴趣区域 / 屏蔽作用 / 结构特征提取 / 特殊形状图像制作
mask = np.zeros(img.shape[:2],np.uint8)  # 创建掩膜
mask[400:600,200:500] = 255   # 制作感兴趣的区域
plt.imshow(mask,cmap=plt.cm.gray)
plt.show()

# 掩膜操作
masked_img = cv.bitwise_and(img,img,mask = mask)
plt.imshow(masked_img,cmap=plt.cm.gray)
plt.show()

# 获取区域直方图
mask_hist = cv.calcHist([img],[0],mask,[256],[0,256])
plt.plot(mask_hist)
plt.show()

# 直方图均衡化，提高图像整体的对比度
# X光图像，提高骨架结构的限时；曝光过度或不足的图像中更好地突出细节
plt.imshow(img,cmap=plt.cm.gray)
plt.show()

dst = cv.equalizeHist(img)
plt.imshow(dst,cmap=plt.cm.gray)
plt.show()

# 自适应的直方图均衡化
# 整个图像被分成很多小块，这些小块被称为“tiles”，然后再对每一个小块分别进行直方图均衡化。 所以在每一个的区域中，直方图会集中在某一个小的区域中
# 如果有噪声的话，噪声会被放大。为了避免这种情况的出现要使用对比度限制。对于每个小块来说，如果直方图中的 bin 超过对比度的上限的话，就把 其中的像素点均匀分散到其他 bins 中，然后在进行直方图均衡化
# 最后，为了去除每一个小块之间的边界，再使用双线性差值，对每一小块进行拼接。

# 创建一个自适应均衡化的对象
cl = cv.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
# 并应用于图像
clahe = cl.apply(img)
plt.imshow(clahe,cmap=plt.cm.gray)
plt.show()
