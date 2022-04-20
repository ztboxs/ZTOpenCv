# USAGE
# python opencv_haar_cascades.py --cascades cascades

# haarcascade_frontalface_default.xml：检测面部
# haarcascade_eye.xml：检测左眼和右眼
# haarcascade_smile.xml：检测面部是否存在嘴部
# haarcascade_eye_tree_eyeglasses.xml：检测是否带墨镜
# haarcascade_frontalcatface.xml：检测猫脸
# haarcascade_frontalcatface_extended.xml：检测猫脸延伸
# haarcascade_frontalface_alt.xml：检测猫脸属性
# haarcascade_frontalface_alt_tree.xml
# haarcascade_frontalface_alt2.xml
# haarcascade_fullbody.xml：检测全身
# haarcascade_lefteye_2splits.xml：检测左眼
# haarcascade_licence_plate_rus_16stages.xml：检测证件
# haarcascade_lowerbody.xml：检测下半身
# haarcascade_profileface.xml
# haarcascade_righteye_2splits.xml：检测右眼
# haarcascade_russian_plate_number.xml：检测俄罗斯字母车牌号
# haarcascade_upperbody.xml：检测上半身

# 导入必要的包
import argparse
import os  # 不同系统路径分隔符
import time  # sleep 2秒

import cv2  # opencv绑定
import imutils
from imutils.video import VideoStream  # 访问网络摄像头

# 构建命令行参数及解析
# --cascades 级联检测器的路径
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascades", type=str, default="cascades",
                help="path to input directory containing haar cascades")
args = vars(ap.parse_args())

# 初始化字典，并保存haar级联检测器名称及文件路径(这里是树莓派存放目录的地址)
face = cv2.CascadeClassifier('F:\ALIIBABA\python3.7\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
eyes = cv2.CascadeClassifier('F:\ALIIBABA\python3.7\Lib\site-packages\cv2\data\haarcascade_eye.xml')
smile = cv2.CascadeClassifier('F:\ALIIBABA\python3.7\Lib\site-packages\cv2\data\haarcascade_smile.xml')
# 初始化视频流，允许摄像头预热2s
print("摄像头正在开启...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# 遍历视频流的每一帧
while True:
    # 获取视频流的每一帧，缩放，并转换灰度图
    frame = vs.read()
    frame = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 使用合适的Haar检测器执行面部检测
    faceRects = face.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=20, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)

    # 遍历检测到的所有面部
    for (fX, fY, fW, fH) in faceRects:
        # 提取面部ROI
        faceROI = gray[fY:fY + fH, fX:fX + fW]

        # 在面部ROI应用左右眼级联检测器
        eyeRects = eyes.detectMultiScale(
            faceROI, scaleFactor=1.1, minNeighbors=25,
            minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE)

        # 在面部ROI应用嘴部检测
        smileRects = smile.detectMultiScale(
            faceROI, scaleFactor=1.1, minNeighbors=25,
            minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE)

        # 遍历眼睛边界框
        for (eX, eY, eW, eH) in eyeRects:
            # 绘制眼睛边界框（红色）
            ptA = (fX + eX, fY + eY)
            ptB = (fX + eX + eW, fY + eY + eH)
            cv2.rectangle(frame, ptA, ptB, (0, 0, 255), 2)

        # 遍历嘴部边界框
        for (sX, sY, sW, sH) in smileRects:
            # 绘制嘴边界框（蓝色）
            ptA = (fX + sX, fY + sY)
            ptB = (fX + sX + sW, fY + sY + sH)
            cv2.rectangle(frame, ptA, ptB, (255, 0, 0), 2)

        # 绘制面部边界框（绿色）
        cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH),
                      (0, 255, 0), 2)

    # 展示输出帧
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # 按下‘q’键，退出循环
    if key == ord("q"):
        break

# 清理工作
cv2.destroyAllWindows()
vs.stop()
