import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class PlantMultimodalDataset(Dataset):
    def __init__(self, root_dir, window_size=250):
        self.root_dir = root_dir
        self.window_size = window_size
        # 严格对齐分类映射
        self.label_map = {"normal": 0, "touch": 1, "light": 2, "stress": 3}
        self.samples = []

        # 1. 统计全局 Z-Score 参数 (Mean, Std)
        self.imp_mean, self.imp_std = self._get_global_imp_stats(root_dir)

        # 2. 加载并切分所有数据
        self._load_data()

    def _get_matched_cols(self, df):
        """核心修复：动态模糊匹配列名，兼容所有的单片机与合成数据"""
        v_col = [c for c in df.columns if '电压' in c]
        imp_col = [c for c in df.columns if '阻抗' in c or '幅值' in c]
        return v_col[0] if v_col else None, imp_col[0] if imp_col else None

    def _get_global_imp_stats(self, root_dir):
        """遍历所有文件，计算真实阻抗的均值(Mean)和标准差(Std)"""
        all_vals = []
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith('.csv'):
                    file_path = os.path.join(root, file)
                    try:
                        df = pd.read_csv(file_path)
                        _, imp_col = self._get_matched_cols(df)
                        if imp_col:
                            # 过滤掉 NaN 空值
                            all_vals.extend(df[imp_col].dropna().tolist())
                    except Exception as e:
                        print(f"⚠️ 读取 {file_path} 统计阻抗时出错: {e}")

        if not all_vals:
            print("🚨 致命警告：没有提取到任何有效的阻抗数据！将使用默认值。")
            return 5900.0, 1.0
            
        mean_val = np.mean(all_vals)
        std_val = np.std(all_vals)
        
        # 防止方差为0导致的除零错误
        if std_val == 0:
            std_val = 1.0
            
        print(f"📊 阻抗 Z-Score 统计完成: Mean={mean_val:.2f}, Std={std_val:.2f}")
        return mean_val, std_val

    def _load_data(self):
        """遍历目录，读取数据并按 window_size 切片"""
        for category in self.label_map.keys():
            cat_dir = os.path.join(self.root_dir, category)
            if not os.path.exists(cat_dir):
                continue

            label = self.label_map[category]
            for file in os.listdir(cat_dir):
                if file.endswith('.csv'):
                    file_path = os.path.join(cat_dir, file)
                    try:
                        df = pd.read_csv(file_path)
                        v_col, imp_col = self._get_matched_cols(df)

                        if not v_col or not imp_col:
                            print(f"⚠️ 跳过 {file}: 找不到电压或阻抗列。")
                            continue

                        volts = df[v_col].values
                        imps = df[imp_col].values

                        # 按窗口大小(默认250)切片
                        num_windows = len(volts) // self.window_size
                        for i in range(num_windows):
                            start_idx = i * self.window_size
                            end_idx = start_idx + self.window_size

                            v_window = volts[start_idx:end_idx]
                            # 取这段窗口内阻抗的均值作为标量特征
                            i_scalar = np.mean(imps[start_idx:end_idx])

                            self.samples.append({
                                'volt': v_window,
                                'imp': i_scalar,
                                'label': label
                            })
                    except Exception as e:
                        print(f"⚠️ 读取 {file_path} 构建样本时出错: {e}")
                        
        print(f"✅ 数据加载与切片完成，共提取到 {len(self.samples)} 个有效样本。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 1. 处理电压特征: [Channel=1, SeqLen=250]
        volt_tensor = torch.FloatTensor(sample['volt']).unsqueeze(0)

        # 2. 处理阻抗特征: Z-Score 标准化
        norm_imp = (sample['imp'] - self.imp_mean) / self.imp_std
        imp_tensor = torch.FloatTensor([norm_imp])

        # 3. 标签
        label = torch.tensor(sample['label'], dtype=torch.long)

        return volt_tensor, imp_tensor, label