import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.font_manager import FontProperties

def analyze_performance():
    image_names = ['kodim08.png', 'kodim19.png', 'kodim20.png', 'kodim21.png', 'kodim22.png', 'kodim24.png']
    student_info = "学号: 230162402007 姓名: 巩怡"
    
    # 加载中文字体 
    my_font = FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc')
    
    # 扩展字典，增加 ORB
    metrics = {
        'SIFT': {'time': [], 'count': []},
        'Harris': {'time': [], 'count': []},
        'KLT': {'time': [], 'count': []},
        'ORB': {'time': [], 'count': []}
    }

    # 初始化检测器
    sift = cv2.SIFT_create()
    orb = cv2.ORB_create(nfeatures=1000) # 设定最大特征点数为 1000 以便公平对比

    for img_name in image_names:
        if not os.path.exists(img_name):
            print(f"⚠️ 跳过缺失的图片: {img_name}")
            continue

        img = cv2.imread(img_name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_float32 = np.float32(gray)

        # 测试 1: SIFT
        t1 = time.time()
        kp_sift, _ = sift.detectAndCompute(gray, None)
        metrics['SIFT']['time'].append((time.time() - t1) * 1000)
        metrics['SIFT']['count'].append(len(kp_sift))

        # 测试 2: Harris
        t2 = time.time()
        dst = cv2.cornerHarris(gray_float32, 2, 3, 0.04)
        h_count = np.sum(dst > 0.01 * dst.max())
        metrics['Harris']['time'].append((time.time() - t2) * 1000)
        metrics['Harris']['count'].append(h_count)

        # 测试 3: KLT (Shi-Tomasi)
        t3 = time.time()
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=1000, qualityLevel=0.01, minDistance=10)
        k_count = len(corners) if corners is not None else 0
        metrics['KLT']['time'].append((time.time() - t3) * 1000)
        metrics['KLT']['count'].append(k_count)

        # 测试 4: ORB (新增)
        t4 = time.time()
        kp_orb, _ = orb.detectAndCompute(gray, None)
        metrics['ORB']['time'].append((time.time() - t4) * 1000)
        metrics['ORB']['count'].append(len(kp_orb))

    algorithms = ['SIFT', 'Harris', 'KLT', 'ORB']
    avg_times = [np.mean(metrics[a]['time']) for a in algorithms]
    avg_counts = [np.mean(metrics[a]['count']) for a in algorithms]

    # 打印总结
    print("\n【Benchmark 性能总结】")
    for i, algo in enumerate(algorithms):
        print(f"{algo:>8}: {avg_times[i]:>6.2f} ms | {avg_counts[i]:>5.0f} pts")

    # 绘图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3']

    # 时间对比
    bars1 = ax1.bar(algorithms, avg_times, color=colors)
    ax1.set_title('Avg Time (ms) - Lower is Faster')
    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.1f}', ha='center', va='bottom')

    # 数量对比
    bars2 = ax2.bar(algorithms, avg_counts, color=colors)
    ax2.set_title('Avg Keypoints Detected')
    for bar in bars2:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{int(bar.get_height())}', ha='center', va='bottom')

    # 添加页脚信息
    fig.text(0.5, 0.02, student_info, ha='center', fontsize=12, fontproperties=my_font)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('Full_Algorithm_Benchmark.png', bbox_inches='tight', dpi=150)
    plt.show()

if __name__ == "__main__":
    analyze_performance()