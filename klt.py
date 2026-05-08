import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.font_manager import FontProperties

def run_klt_detection():
    print("--- 开始执行 KLT (Shi-Tomasi) 角点检测 (完美中文显示版) ---")
    
    # 1. 定义图片列表和学号姓名
    image_names = ['kodim08.png', 'kodim19.png', 'kodim20.png', 'kodim21.png', 'kodim22.png', 'kodim24.png']
    student_info = "学号: 230162402007 姓名: 巩怡"

    # 2. 加载 Linux 系统中的开源中文字体 (文泉驿正黑)
    # 如果该路径报错，请确认是否已执行 sudo apt-get install fonts-wqy-zenhei -y
    my_font = FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')

    for img_name in image_names:
        if not os.path.exists(img_name):
            print(f"⚠️ 找不到图片 {img_name}，已跳过。")
            continue

        # 3. 读取图片并转换为灰度图
        img = cv2.imread(img_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 4. 执行 KLT (Shi-Tomasi) 特征检测
        # 参数: 灰度图, 最大角点数=500, 质量水平=0.01, 角点最小欧式距离=10
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=500, qualityLevel=0.01, minDistance=10)

        img_result = img.copy()

        # 5. 绘制检测到的角点 (不在原图数据内写字)
        if corners is not None:
            corners = np.int32(corners)
            for i in corners:
                x, y = i.ravel()
                # 绘制绿色圆圈
                cv2.circle(img_result, (x, y), 3, (0, 255, 0), -1)

        # 6. 转换为 RGB 供 Matplotlib 显示
        img_rgb = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
        
        # 7. 使用 Matplotlib 控制布局，添加外部文字
        fig, ax = plt.subplots(figsize=(10, 7)) 
        ax.imshow(img_rgb)
        ax.axis('off') 
        ax.set_title(f"KLT (Shi-Tomasi) Detection - {img_name}")

        # 在画布底部居中添加学号姓名，使用 fontproperties 指定中文字体
        # y=0.05 将文字放置在图片下方的留白区域
        fig.text(0.5, 0.05, student_info, ha='center', fontsize=16, fontproperties=my_font)

        # 8. 保存含有外部文字的完整布局
        output_filename = f"External_KLT_result_{img_name}"
        plt.savefig(output_filename, bbox_inches='tight', dpi=150)
        print(f"已处理并保存：{output_filename}")
        
        # 清理当前画布，准备下一张图片
        plt.close(fig) 

if __name__ == "__main__":
    run_klt_detection()