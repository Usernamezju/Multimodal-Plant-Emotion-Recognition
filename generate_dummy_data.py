import numpy as np
import pandas as pd
import os
from tqdm import tqdm

# --- 核心参数设置 ---
FS = 250               # 采样率 (Hz)
DURATION = 5           # 每个样本的时长 (秒) -> 1250 个数据点
NUM_SAMPLES = 200      # 每个类别生成的样本数量 (4类共 800 个文件)
OUTPUT_DIR = "dataset_synthetic"

# --- 基础生理基准值 (基于你上传的参考数据) ---
BASE_VOLTAGE = -2.5    # 基准电压 (V)
BASE_IMPEDANCE = 5900  # 基准高频阻抗
NOISE_VOLT_STD = 0.2   # 电压白噪声强度
NOISE_IMP_STD = 5.0    # 阻抗白噪声强度

def generate_base_signals(t):
    """生成包含白噪声和 50Hz 工频干扰的基底信号"""
    # 电压：基线 + 白噪声 + 50Hz工频(模拟环境中未完全屏蔽的电网干扰)
    v_noise = np.random.normal(0, NOISE_VOLT_STD, len(t))
    v_powerline = 0.5 * np.sin(2 * np.pi * 50 * t) 
    volt = np.full_like(t, BASE_VOLTAGE) + v_noise + v_powerline
    
    # 阻抗：基线 + 微小白噪声
    imp = np.full_like(t, BASE_IMPEDANCE) + np.random.normal(0, NOISE_IMP_STD, len(t))
    return volt, imp

def generate_sample(category, t):
    """根据类别注入特定的生理学响应波形"""
    volt, imp = generate_base_signals(t)
    
    # 刺激发生的时间点 (1.0秒 到 2.0秒之间随机)
    stimulus_t = np.random.uniform(1.0, 2.0)
    stimulus_idx = int(stimulus_t * FS)
    
    if category == "normal":
        # 正常状态：只有基底波动，随时间可能发生极缓慢的几欧姆的随机游走
        imp += np.linspace(0, np.random.uniform(-10, 10), len(t))
        
    elif category == "touch":
        # 触摸响应 (Fast Phase)
        # 1. 电压：动作电位(AP)尖峰 -> 极快上升，指数级回落
        ap_amplitude = np.random.uniform(1.5, 3.0) # 瞬时跳变 1.5~3V
        decay_v = np.exp(-(t[stimulus_idx:] - stimulus_t) * 10) # 快速衰减
        volt[stimulus_idx:] += ap_amplitude * decay_v
        
        # 2. 阻抗：气孔响应 -> 延迟 0.2 秒后，阻抗发生下凹
        imp_delay_idx = int((stimulus_t + 0.2) * FS)
        if imp_delay_idx < len(t):
            imp_drop = np.random.uniform(-100, -250)
            decay_i = np.exp(-(t[imp_delay_idx:] - (stimulus_t + 0.2)) * 2) # 恢复较慢
            # 制造一个下凹再恢复的波形 (用正弦或负指数混合，这里用简化的下降-恢复)
            imp[imp_delay_idx:] += imp_drop * (1 - decay_i) * np.exp(-(t[imp_delay_idx:] - (stimulus_t + 0.2)) * 1)

    elif category == "light":
        # 光照变化响应 (Slow Phase)
        # 1. 电压：变异电位(VP) -> 长时期缓慢漂移
        vp_drift = np.random.uniform(-1.0, -2.0)
        # Sigmoid 平滑过渡模拟缓慢的光合作用电位变化
        drift_curve = vp_drift / (1 + np.exp(-3 * (t[stimulus_idx:] - stimulus_t - 1.0)))
        volt[stimulus_idx:] += drift_curve
        
        # 2. 阻抗：非常微弱的单调上升 (水分微量蒸腾)
        imp[stimulus_idx:] += np.linspace(0, np.random.uniform(20, 50), len(t) - stimulus_idx)

    elif category == "stress":
        # 机械胁迫/损伤 (Mixed & Permanent Phase)
        # 1. 电压：尖峰 + 永久性基线偏移
        ap_amplitude = np.random.uniform(2.0, 4.0)
        permanent_offset = np.random.uniform(1.0, 2.0)
        decay_v = np.exp(-(t[stimulus_idx:] - stimulus_t) * 5)
        volt[stimulus_idx:] += (ap_amplitude * decay_v) + permanent_offset
        
        # 2. 阻抗：细胞液流失导致阻抗呈现不可逆的阶跃上升
        imp_jump = np.random.uniform(500, 1000)
        # 快速上升后保持高位
        jump_curve = imp_jump * (1 - np.exp(-(t[stimulus_idx:] - stimulus_t) * 5))
        imp[stimulus_idx:] += jump_curve

    return volt, imp

def main():
    categories = ["normal", "touch", "light", "stress"]
    t = np.linspace(0, DURATION, int(FS * DURATION), endpoint=False)
    
    print(f"🌱 开始生成植物生理级合成数据 (共 {len(categories) * NUM_SAMPLES} 个样本)...")
    
    for cat in categories:
        cat_dir = os.path.join(OUTPUT_DIR, cat)
        os.makedirs(cat_dir, exist_ok=True)
        
        for i in tqdm(range(NUM_SAMPLES), desc=f"Generating '{cat}' data"):
            # 生成生理信号
            volt, imp = generate_sample(cat, t)
            
            # 构造 DataFrame，完全对齐真实单片机的 CSV 列名
            df = pd.DataFrame({
                "时间(s)": t,
                "电压(V)": volt,
                "幅值(高频80K)": imp
            })
            
            # 保存为 CSV
            file_path = os.path.join(cat_dir, f"synth_{cat}_{i:04d}.csv")
            df.to_csv(file_path, index=False)
            
    print(f"✅ 生成完毕！所有数据已保存在 '{OUTPUT_DIR}' 目录下。")

if __name__ == "__main__":
    main()