import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
import os

def analyze_performance():
    print("--- 开始执行算法性能横向对比分析 ---")
    
    image_names = ['kodim08.png', 'kodim19.png', 'kodim20.png', 'kodim21.png', 'kodim22.png', 'kodim24.png']
    
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

        start_time = time.time()
        keypoints, _ = sift.detectAndCompute(gray, None)
        end_time = time.time()
        
        metrics['SIFT']['time'].append((end_time - start_time) * 1000)
        metrics['SIFT']['count'].append(len(keypoints))

        start_time = time.time()
        dst = cv2.cornerHarris(gray_float32, 2, 3, 0.04)
        harris_count = np.sum(dst > 0.01 * dst.max())
        end_time = time.time()
        
        metrics['Harris']['time'].append((end_time - start_time) * 1000)
        metrics['Harris']['count'].append(harris_count)

        start_time = time.time()
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=1000, qualityLevel=0.01, minDistance=10)
        klt_count = len(corners) if corners is not None else 0
        end_time = time.time()
        
        metrics['KLT']['time'].append((end_time - start_time) * 1000)
        metrics['KLT']['count'].append(klt_count)

    algorithms = ['SIFT', 'Harris', 'KLT (Shi-Tomasi)']
    avg_times = [np.mean(metrics['SIFT']['time']), np.mean(metrics['Harris']['time']), np.mean(metrics['KLT']['time'])]
    avg_counts = [np.mean(metrics['SIFT']['count']), np.mean(metrics['Harris']['count']), np.mean(metrics['KLT']['count'])]

    print("\n【性能分析总结 (所有图片的平均值)】")
    for i, algo in enumerate(algorithms):
        print(f"{algo:>16}: 平均耗时 {avg_times[i]:.2f} ms | 平均检测到 {avg_counts[i]:.0f} 个特征点")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bars1 = ax1.bar(algorithms, avg_times, color=['#4C72B0', '#55A868', '#C44E52'])
    ax1.set_title('Average Execution Time (Lower is Faster)')
    ax1.set_ylabel('Time (ms)')
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 1, round(yval, 1), ha='center', va='bottom')

    bars2 = ax2.bar(algorithms, avg_counts, color=['#4C72B0', '#55A868', '#C44E52'])
    ax2.set_title('Average Number of Keypoints Detected')
    ax2.set_ylabel('Count')
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 1, round(yval, 0), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('Algorithm_Performance_Comparison.png')
    plt.show()

if __name__ == "__main__":
    analyze_performance()