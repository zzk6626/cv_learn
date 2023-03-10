import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# 1.获取图像
cap = cv.VideoCapture('./image/DOG.wmv')

# 2.获取第一帧图像，并指定目标位置
ret,frame = cap.read()  # (480, 444, 3)

# 2.1 目标位置（行，高，列，宽）  行 + 高 == 宽度位置  列 + 宽 == 长度位置
r,h,c,w = 197,141,0,208
track_window = (c,r,w,h)
# 2.2 指定目标的感兴趣区域
roi = frame[r:r+h, c:c+w]

# 3. 计算  感兴趣区域的  直方图
# 3.1 转换色彩空间（HSV）
hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)  # HSV类似于BGR shape(141, 208, 3)

# 3.2 去除低亮度的值
# mask = cv.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))

# 3.3 计算直方图
roi_hist = cv.calcHist([hsv_roi],[0],None,[180],[0,180])

# 3.4 归一化  max: 9738.0 => 255.0
cv.normalize(roi_hist,roi_hist,0,255,cv.NORM_MINMAX)

# 4. 目标追踪
# 4.1 设置窗口搜索终止条件：最大迭代次数，窗口中心漂移最小值
term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

while(cap.isOpened()):
    # 4.2 获取每一帧图像
    ret,frame = cap.read()
    if ret == True:
        # 4.3 计算直方图的反向投影
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)  # 获得每一帧的HSV图像
        dst = cv.calcBackProject([hsv],[0],roi_hist,[0,180],1)   # [0]通道,组距为1，产生的和原图像大小一样的反向投影矩阵，
        print(dst.shape)

        # 4.4 进行meanshift追踪track_window，直接将目标设置为原图像的一部分，并未采用灰度直方图，后期匹配
        ret, track_window = cv.meanShift(dst, track_window, term_crit)

        # 4.5 将追踪的位置绘制在视频上，并进行显示track_window
        x,y,w,h = track_window
        img2 = cv.rectangle(frame, (x,y), (x+w,y+h),(255,0,0),2)
        cv.imshow('frame',img2)

        if cv.waitKey(60) & 0xFF == ord('q'):
            break
    else:
        break
# 5. 资源释放
cap.release()
cv.destroyAllWindows()