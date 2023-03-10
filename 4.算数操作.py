import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


# Numpy的添加是模运算，Opencv加法是饱和操作,一般相加推荐Opencv，图片更加饱和.
# 要求两幅图像大小相同
x = np.uint8([250])
y = np.uint8([10])
print(cv.add(x,y)) # 250 + 10 = 260 ==> 255
print(x+y) # 260 % 256 = 4

# 减法
print(cv.subtract(y,x))   # -240 ==> 0
print(y - x)  # 240 % 256 ==> 16

img1 = cv.imread('data/1.jpg')
print(img1.shape)
img2 = np.zeros(img1.shape,np.uint8)
plt.imshow(img2[:,:,::-1])
plt.show()

img3 = cv.add(img1,img2)
plt.imshow(img3[:,:,::-1])
plt.show()

img4 = img1 + img2
plt.imshow(img4[:,:,::-1])
plt.show()

# 图像的混合 -- 其实也是加法，但是两幅图像的权重不同，会给人一种混合或者透明的感觉
# dst = a * img1 + b * img2 + c ==>  一般情况下 a + b = 1.0
img3 = cv.addWeighted(img1,0.7,img2,0.3,0)  # a = 0.7 ,b = 0.3,c = 0
plt.imshow(img3[:,:,::-1])
plt.show()