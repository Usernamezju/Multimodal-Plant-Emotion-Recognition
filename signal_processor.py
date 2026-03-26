import numpy as np
import pandas as pd
import os
from scipy.signal import iirnotch, butter, filtfilt
import matplotlib.pyplot as plt
from tqdm import tqdm

class PlantSignalFilter:
    def __init__(self, fs=250.0):
        """
        初始化滤波器
        :param fs: 采样频率 (Hz)，根据单片机 CSV 数据默认为 250Hz
        """
        self.fs = fs
        self.nyq = self.fs / 2.0  # 奈奎斯特频率

    def notch_filter(self, data, freq=50.0, q=30.0):
        """
        50Hz 工频陷波器
        :param freq: 需要滤除的中心频率 (默认 50Hz)
        :param q: 品质因数，控制陷波的带宽 (值越大，滤除的频带越窄)
        """
        w0 = freq / self.nyq
        b, a = iirnotch(w0, q)
        # 使用 filtfilt 进行零相位滤波，防止信号波形发生时间偏移
        return filtfilt(b, a, data)

    def lowpass_filter(self, data, cutoff=20.0, order=4):
        """
        巴特沃斯低通滤波器
        :param cutoff: 截止频率 (Hz)，植物信号多为低频，暂定 20Hz 保留瞬态特征
        :param order: 滤波器的阶数
        """
        normal_cutoff = cutoff / self.nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, data)

    def process(self, data):
        """
        级联执行滤波主流程
        """
        # 1. 先去除最致命的 50Hz 工频干扰
        notched_data = self.notch_filter(data)
        # 2. 再滤除高频毛刺与白噪声
        smoothed_data = self.lowpass_filter(notched_data)

        # 专利中的参数映射：归一化到 [-1, 1] 附近，并消除直流偏置
        mean_val = np.mean(smoothed_data)
        std_val = np.std(smoothed_data) + 1e-8
        return (smoothed_data - mean_val) / std_val
    
def batch_process_dataset(input_root, output_root):
    processor = PlantSignalFilter(fs=250.0)
    
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # 遍历 dataset 下的所有分类文件夹 (touch, light, normal 等)
    categories = [d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))]
    
    for cat in categories:
        in_cat_path = os.path.join(input_root, cat)
        out_cat_path = os.path.join(output_root, cat)
        
        if not os.path.exists(out_cat_path):
            os.makedirs(out_cat_path)
            
        print(f"🚀 正在快速处理类别: {cat}")
        files = [f for f in os.listdir(in_cat_path) if f.endswith('.csv')]
        
        for file_name in tqdm(files):
            file_path = os.path.join(in_cat_path, file_name)
            try:
                # 读取数据 (跳过可能存在的阻抗文件，只处理电压数据)
                df = pd.read_csv(file_path)
                if '电压' in df.columns:
                    raw_v = df['电压'].values
                    # 执行快速滤波
                    filtered_v = processor.process(raw_v)
                    df['电压'] = filtered_v
                
                # 写回结果
                df.to_csv(os.path.join(out_cat_path, file_name), index=False)
            except Exception as e:
                print(f"跳过文件 {file_name}: {e}")

# === 测试模块 ===
if __name__ == "__main__":
    # df = pd.read_csv('/home/yuki_noa/plant_condition_model/dataset/light/26-03-07 15_30_24_915.csv') 
    # raw_voltage = df['电压(V)'].values if '电压(V)' in df.columns else df['幅值(高频80K)'].values
    # t = df['时间(s)'].values
    
    # # 实例化并处理
    # signal_processor = PlantSignalFilter(fs=250.0)
    # filtered_voltage = signal_processor.process(raw_voltage)

    # 可视化对比(可选)
    # plt.figure(figsize=(10, 6))
    # plt.plot(t, raw_voltage, label='Raw Signal (with noise)', alpha=0.6)
    # plt.plot(t, filtered_voltage, label='Filtered Signal', linewidth=2)
    # plt.title('Plant Electrical Signal Preprocessing')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Voltage (V) / 幅值')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # 配置路径
    INPUT_DIR = 'dataset_real_condition'
    OUTPUT_DIR = INPUT_DIR + '_filtered'
    
    batch_process_dataset(INPUT_DIR, OUTPUT_DIR)
    print(f"\n✅ 滤波任务已完成！数据已存入 {OUTPUT_DIR} 文件夹。")