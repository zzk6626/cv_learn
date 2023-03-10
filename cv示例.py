import cv2
# 读取一个图片
lena = cv2.imread('F:/Pytorch/data/1.jpg',-1)   # 读取图像,1--彩色模式，0--灰度模式，-1--alpha通道，透明度
cv2.imshow('image',lena)
cv2.waitKey(0)
