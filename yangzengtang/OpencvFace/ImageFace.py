# USAGE
# python opencv_haar_cascades_images.py --cascades cascades --image ml.jpg

# 导入必要的包
import argparse
import cv2  # opencv绑定
import imutils

# 构建命令行参数及解析
# --cascades 级联检测器的路径
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascades", type=str, default="cascades",
                help="path to input directory containing haar cascades")
# default--图片路径(可以绝对路径和相对路径)
ap.add_argument("-i", "--image", type=str, default="hou3.png",
                help="path to input image")
args = vars(ap.parse_args())
# 初始化字典，并保存haar级联检测器名称及文件路径(这里是树莓派(pc)存放目录的地址)
face = cv2.CascadeClassifier('F:\ALIIBABA\python3.7\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
eyes = cv2.CascadeClassifier('F:\ALIIBABA\python3.7\Lib\site-packages\cv2\data\haarcascade_eye.xml')
smile = cv2.CascadeClassifier('F:\ALIIBABA\python3.7\Lib\site-packages\cv2\data\haarcascade_smile.xml')

# 从磁盘读取图像，缩放，并转换灰度图
print(args['image'])
image = cv2.imread(args["image"])
image = imutils.resize(image, width=600)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用合适的Haar检测器执行面部检测
faceRects = face.detectMultiScale(
    gray, scaleFactor=1.05, minNeighbors=1, minSize=(300, 300),
    flags=cv2.CASCADE_SCALE_IMAGE)

# 遍历检测到的所有面部
for (fX, fY, fW, fH) in faceRects:
    # 提取面部ROI
    faceROI = gray[fY:fY + fH, fX:fX + fW]

    # 在面部ROI应用左右眼级联检测器
    eyeRects = eyes.detectMultiScale(
        faceROI, scaleFactor=1.1, minNeighbors=65,
        minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE)

    # 在面部ROI应用嘴部检测
    smileRects = smile.detectMultiScale(
        faceROI, scaleFactor=1.1, minNeighbors=80,
        minSize=(15, 15), flags=cv2.CASCADE_SCALE_IMAGE)

    # 遍历眼睛边界框
    for (eX, eY, eW, eH) in eyeRects:
        # 绘制眼睛边界框（红色）
        ptA = (fX + eX, fY + eY)
        ptB = (fX + eX + eW, fY + eY + eH)
        cv2.rectangle(image, ptA, ptB, (0, 0, 255), 2)

    # 遍历嘴部边界框
    for (sX, sY, sW, sH) in smileRects:
        # 绘制嘴边界框（蓝色）
        ptA = (fX + sX, fY + sY)
        ptB = (fX + sX + sW, fY + sY + sH)
        cv2.rectangle(image, ptA, ptB, (255, 0, 0), 2)

    # 绘制面部边界框（绿色）
    cv2.rectangle(image, (fX, fY), (fX + fW, fY + fH),
                  (0, 255, 0), 2)

# 展示输出帧
cv2.imshow("image", image)
cv2.waitKey(0)
# 清理工作
cv2.destroyAllWindows()
