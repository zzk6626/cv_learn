import numpy as np
import cv2
import matplotlib.pyplot as plt
img1 = cv2.imread('data/1.jpg',1)
print(img1.shape)   # (818, 828, 3)

cv2.imshow('image',img1)  # 先创建窗口，改变窗口大小可以使用cv2.nameWindow()---cv2.namedWindow('messi', cv2.WINDOW_AUTOSIZE)
cv2.waitKey(0)  # 图像绘制等待时间，0--永久停留
cv2.destroyAllWindows()  # 删除所有窗口

# matplotlib中显示
plt.imshow(img1[:,:,::-1]) # 图像显示的时候，opencv采用的是BGR，而matplotlib采用的是RGB通道，故通道3需要翻转
plt.show()

img2 = cv2.imread('data/1.jpg',0)
cv2.imwrite('data/lanyangyang.jpg',img2)  # 保存图像
# 显示灰度图像
plt.imshow(img2,cmap=plt.cm.gray)
plt.show()