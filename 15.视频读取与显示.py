import numpy as np
import cv2 as cv
# 1.获取视频对象
cap = cv.VideoCapture('./image/DOG.wmv')
# 2.判断是否读取成功
while(cap.isOpened()):
    # 3.获取每一帧图像
    ret, frame = cap.read()  # 自动迭代，while不断循环，frame不断更新
    # 4. 获取成功显示图像
    if ret == True:
        cv.imshow('frame',frame)
    # 5.每一帧间隔为25ms
    # cv2.waitKey(delay)参数  delay取正整数 等待按键的时间cv2.waitKey(25)，就是等待25（milliseconds),视频中一帧数据显示（停留）的时间
    # cv2.waitKey(delay)返回值：
    # 1、等待期间有按键：返回按键的ASCII码（比如：q的ASCII码为113）
    # 2、等待期间没有按键：返回 -1；

    # 引入& 0xFF==27 那么得出的结果永远是后八位，这样就可以排除其他按键的干扰
    # 当小键盘数字键“NumLock”激活时，“q”对应的ASCII值为100000000000001100011 。
    # 而其他情况下，对应的ASCII值为01100011。
    #  ord('q') 表示q的ASCH码的二进制参数
    if cv.waitKey(50) & 0xFF == ord('q'):
        break
# 6.释放视频对象
cap.release()
cv.destroyAllWindows()