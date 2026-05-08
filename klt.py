import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def run_klt_detection():
    print("--- 开始执行 KLT (Shi-Tomasi) 角点检测 ---")
    # 1. 定义图片列表和学号姓名
    image_names = ['kodim08.png', 'kodim19.png', 'kodim20.png', 'kodim21.png', 'kodim22.png', 'kodim24.png']
    student_info = "学号: 230162402007 姓名: 巩怡"
    for img_name in image_names:
        if not os.path.exists(img_name):
            print(f"⚠️ 找不到图片 {img_name}，已跳过。")
            continue

        # 2. 读取图片并转换为灰度图
        img = cv2.imread(img_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 3. 执行 KLT (Shi-Tomasi) 特征检测
        # 参数: 灰度图, 最大角点数=500, 质量水平=0.01, 角点最小欧式距离=10
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=500, qualityLevel=0.01, minDistance=10)

        img_result = img.copy()

        # 4. 绘制检测到的角点
        if corners is not None:
            # 转换为整型坐标
            corners = np.int32(corners)
            for i in corners:
                x, y = i.ravel()
                # 绘制绿色圆圈 (BGR格式: 0, 255, 0)
                cv2.circle(img_result, (x, y), 3, (0, 255, 0), -1)

        # 5. 在右下角绘制学号姓名
        h, w = img_result.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(student_info, font, 0.8, 2)[0]
        cv2.putText(img_result, student_info, (w - text_size[0] - 15, h - 15), font, 0.8, (0, 0, 255), 2)

        # 6. 单独保存并创建显示窗口
        cv2.imwrite(f"KLT_result_{img_name}", img_result)
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
        plt.title(f"KLT (Shi-Tomasi) Detection - {img_name}")
        plt.axis('off')

    plt.show()

if __name__ == "__main__":
    run_klt_detection()