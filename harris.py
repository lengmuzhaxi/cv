import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def run_harris_detection():
    print("--- 开始执行 Harris 角点检测 ---")
    # 1. 定义图片列表和学号姓名
    image_names = ['kodim08.png', 'kodim19.png', 'kodim20.png', 'kodim21.png', 'kodim22.png', 'kodim24.png']
    student_info = "学号: 230162402007 姓名: 巩怡"

    for img_name in image_names:
        if not os.path.exists(img_name):
            print(f"⚠️ 找不到图片 {img_name}，已跳过。")
            continue

        # 2. 读取图片并转换为 float32 格式的灰度图
        img = cv2.imread(img_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)

        # 3. 执行 Harris 角点检测
        # 参数: 灰度图, 邻域大小 blockSize=2, Sobel算子孔径 ksize=3, Harris参数 k=0.04
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)

        # 膨胀检测结果，以便在图上更清晰地显示角点
        dst = cv2.dilate(dst, None)

        # 创建原图的拷贝用于绘制
        img_result = img.copy()
        
        # 4. 设定阈值，将角点位置标记为红色 (BGR格式: 0, 0, 255)
        # 0.01 * dst.max() 是阈值，可以根据需要微调来增减角点数量
        img_result[dst > 0.01 * dst.max()] = [0, 0, 255]
        h, w = img_result.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(student_info, font, 0.8, 2)[0]
        cv2.putText(img_result, student_info, (w - text_size[0] - 15, h - 15), font, 0.8, (0, 0, 255), 2)
        cv2.imwrite(f"Harris_result_{img_name}", img_result)
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
        plt.title(f"Harris Corners - {img_name}")
        plt.axis('off')

    plt.show()

if __name__ == "__main__":
    run_harris_detection()