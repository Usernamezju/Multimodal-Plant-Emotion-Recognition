import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os

class PlantMultimodalDataset(Dataset):
    def __init__(self, root_dir, window_size=250):
        self.samples = []
        self.window_size = window_size
        self.label_map = {"normal": 0, "touch": 1, "light": 2, "stress": 3}
        
        # 1. 统计全集阻抗特征，获取min-max归一化参数
        self.imp_min, self.imp_max, self.imp_mean = self._get_global_imp_stats(root_dir)
        print(f"阻抗统计完成: Min={self.imp_min:.2f}, Max={self.imp_max:.2f}, Mean={self.imp_mean:.2f}")

        # 2. 遍历加载
        for cat_name, label in self.label_map.items():
            cat_path = os.path.join(root_dir, cat_name)
            if os.path.exists(cat_path):
                for file_name in os.listdir(cat_path):
                    if file_name.endswith('.csv'):
                        self._process_file(os.path.join(cat_path, file_name), label)

    def _get_global_imp_stats(self, root_dir):
        """扫描所有类别下的阻抗文件，锁定全局极值"""
        all_vals = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.csv'):
                    df = pd.read_csv(os.path.join(root, file))
                    if '幅值(高频80K)' in df.columns:
                        all_vals.extend(df['幅值(高频80K)'].tolist())
        
        if not all_vals: # 兜底逻辑
            return 5000.0, 6500.0, 5900.0
            
        return np.min(all_vals), np.max(all_vals), np.mean(all_vals)

    def _normalize_imp(self, val):
        """将阻抗映射到 [0, 1] 区间"""
        # 防止分母为0
        denom = self.imp_max - self.imp_min if self.imp_max != self.imp_min else 1.0
        return (val - self.imp_min) / denom

    def _process_file(self, file_path, label):
        df = pd.read_csv(file_path)
        
        """以下代码是针对之前数据缺失时的填充处理策略，即将弃用"""
        # 处理电压 (Light 样本)
        if '电压(V)' in df.columns:
            v_data = df['电压(V)'].values
            for i in range(0, len(v_data) - self.window_size, self.window_size // 2):
                chunk = v_data[i : i + self.window_size]
                self.samples.append({
                    'volt': torch.FloatTensor(chunk).unsqueeze(0),
                    'imp': torch.FloatTensor([self._normalize_imp(self.imp_mean)]), # 用全局均值填充
                    'label': label
                })

        # 处理阻抗 (Normal/Touch 样本)
        if '幅值(高频80K)' in df.columns:
            for val in df['幅值(高频80K)'].values:
                self.samples.append({
                    'volt': torch.zeros(1, self.window_size),
                    'imp': torch.FloatTensor([self._normalize_imp(val)]),
                    'label': label
                })

        # 假设新 CSV 中同时拥有两列
        # v_data = df['电压(V)'].values
        # imp_data = df['幅值(高频80K)'].values

        # # 以 250 点（1秒）的电压为一个窗口
        # for i in range(0, len(v_data) - 250, 125):
        #     volt_chunk = v_data[i : i + 250]
            
        #     # 提取这 1 秒内的阻抗均值作为这一段电压的伴随生理特征
        #     imp_chunk_mean = np.mean(imp_data[i : i + 250])
            
        #     # 打包数据
        #     self.samples.append({
        #         'volt': torch.FloatTensor(volt_chunk).unsqueeze(0),
        #         'imp': torch.FloatTensor([self._normalize_imp(imp_chunk_mean)]),
        #         'label': label
        #     })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s['volt'], s['imp'], s['label']