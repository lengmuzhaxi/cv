import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.font_manager import FontProperties

def analyze_performance():
    print("--- 开始执行算法性能横向对比分析 (含中文信息版) ---")
    
    image_names = ['kodim08.png', 'kodim19.png', 'kodim20.png', 'kodim21.png', 'kodim22.png', 'kodim24.png']
    student_info = "学号: 230162402007 姓名: 巩怡"
    
    # 加载中文字体
    my_font = FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
    
    metrics = {
        'SIFT': {'time': [], 'count': []},
        'Harris': {'time': [], 'count': []},
        'KLT': {'time': [], 'count': []}
    }

    sift = cv2.SIFT_create()

    for img_name in image_names:
        if not os.path.exists(img_name):
            print(f"⚠️ 跳过缺失的图片: {img_name}")
            continue

        img = cv2.imread(img_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_float32 = np.float32(gray)

        # 测试 1: SIFT
        start_time = time.time()
        keypoints, _ = sift.detectAndCompute(gray, None)
        end_time = time.time()
        metrics['SIFT']['time'].append((end_time - start_time) * 1000)
        metrics['SIFT']['count'].append(len(keypoints))

        # 测试 2: Harris
        start_time = time.time()
        dst = cv2.cornerHarris(gray_float32, 2, 3, 0.04)
        harris_count = np.sum(dst > 0.01 * dst.max())
        end_time = time.time()
        metrics['Harris']['time'].append((end_time - start_time) * 1000)
        metrics['Harris']['count'].append(harris_count)

        # 测试 3: KLT
        start_time = time.time()
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=1000, qualityLevel=0.01, minDistance=10)
        klt_count = len(corners) if corners is not None else 0
        end_time = time.time()
        metrics['KLT']['time'].append((end_time - start_time) * 1000)
        metrics['KLT']['count'].append(klt_count)

    algorithms = ['SIFT', 'Harris', 'KLT (Shi-Tomasi)']
    avg_times = [np.mean(metrics['SIFT']['time']), np.mean(metrics['Harris']['time']), np.mean(metrics['KLT']['time'])]
    avg_counts = [np.mean(metrics['SIFT']['count']), np.mean(metrics['Harris']['count']), np.mean(metrics['KLT']['count'])]

    # 打印文字版数据总结
    print("\n【性能分析总结 (所有图片的平均值)】")
    for i, algo in enumerate(algorithms):
        print(f"{algo:>16}: 平均耗时 {avg_times[i]:.2f} ms | 平均检测到 {avg_counts[i]:.0f} 个特征点")

    # 绘制性能对比图表
    # 增加 figsize 的高度（从 5 改为 6），为底部文字留出空间
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # 子图 1：处理时间对比
    bars1 = ax1.bar(algorithms, avg_times, color=['#4C72B0', '#55A868', '#C44E52'])
    ax1.set_title('Average Execution Time (Lower is Faster)')
    ax1.set_ylabel('Time (ms)')
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 1, round(yval, 1), ha='center', va='bottom')

    # 子图 2：特征点数量对比
    bars2 = ax2.bar(algorithms, avg_counts, color=['#4C72B0', '#55A868', '#C44E52'])
    ax2.set_title('Average Number of Keypoints Detected')
    ax2.set_ylabel('Count')
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 1, round(yval, 0), ha='center', va='bottom')
    fig.text(0.5, 0.02, student_info, ha='center', fontsize=14, fontproperties=my_font)

    plt.tight_layout(rect=[0, 0.05, 1, 1]) # 调整布局，防止底部文字被切掉
    plt.savefig('Algorithm_Performance_Comparison_with_ID.png', bbox_inches='tight', dpi=150)
    print(f"性能分析对比图已保存为：Algorithm_Performance_Comparison_with_ID.png")
    plt.show()

if __name__ == "__main__":
    analyze_performance()