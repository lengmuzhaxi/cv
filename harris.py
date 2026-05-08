import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.font_manager import FontProperties

def run_harris_detection():
    print("--- 开始执行 Harris 角点检测 (完美中文显示版) ---")
    
    # 1. 定义图片列表和学号姓名
    image_names = ['kodim08.png', 'kodim19.png', 'kodim20.png', 'kodim21.png', 'kodim22.png', 'kodim24.png']
    student_info = "学号: 230162402007 姓名: 巩怡"

    # 2. 加载在 Linux 系统中安装的开源中文字体 (文泉驿正黑)
    my_font = FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')

    for img_name in image_names:
        if not os.path.exists(img_name):
            print(f"⚠️ 找不到图片 {img_name}，已跳过。")
            continue

        # 3. 读取图片并转换为 float32 格式的灰度图
        img = cv2.imread(img_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)

        # 4. 执行 Harris 角点检测
        # 参数: 灰度图, 邻域大小 blockSize=2, Sobel算子孔径 ksize=3, Harris参数 k=0.04
        dst = cv2.cornerHarris(gray, 2, 3, 0.04)

        # 膨胀检测结果，以便在图上更清晰地显示角点
        dst = cv2.dilate(dst, None)

        # 创建原图的拷贝用于绘制
        img_result = img.copy()
        
        # 5. 设定阈值，将角点位置标记为红色
        img_result[dst > 0.01 * dst.max()] = [0, 0, 255]

        # 6. 转换为 RGB 供 Matplotlib 显示
        img_rgb = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
        
        # 7. 使用 Matplotlib 控制布局，添加外部文字
        fig, ax = plt.subplots(figsize=(10, 7)) 
        ax.imshow(img_rgb)
        ax.axis('off') # 隐藏坐标轴
        ax.set_title(f"Harris Corners - {img_name}")

        # 在画布底部居中添加学号姓名，并使用 fontproperties 指定中文字体
        fig.text(0.5, 0.05, student_info, ha='center', fontsize=16, fontproperties=my_font)

        # 8. 保存含有外部文字的完整布局
        output_filename = f"External_Harris_result_{img_name}"
        plt.savefig(output_filename, bbox_inches='tight', dpi=150)
        print(f"已处理并保存：{output_filename}")
        
        # 清理当前画布，准备下一张图片，防止内存溢出
        plt.close(fig)

if __name__ == "__main__":
    run_harris_detection()