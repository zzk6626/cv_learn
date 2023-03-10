import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
# 1 图像读取
img = cv.imread('./image/tv.jpg')

# 2 ORB角点检测
# 2.1 实例化ORB对象
orb = cv.ORB_create(nfeatures=500)
# 2.2 检测关键点,并计算特征描述符
kp,des = orb.detectAndCompute(img,None)

print(des.max())
print(des.shape)  # 可以看出shape(32,),最大值255（2^8-1）   32 * 8 (0/1) = 256位0/1描述符

# 3 将关键点绘制在图像上
img2 = cv.drawKeypoints(img, kp, None, color=(0,0,255), flags=0)

# 4. 绘制图像
plt.figure(figsize=(10,8),dpi=100)
plt.imshow(img2[:,:,::-1])
plt.xticks([]), plt.yticks([])
plt.show()