本项目基于多模态融合技术与自主研发的 **双相激活函数 (PBA)**，旨在通过植物电位 (Electrical Potential) 与复阻抗 (Impedance) 信号，识别植物的生理情绪状态（如：正常、触碰、光照应激、环境胁迫）。

## 📁 目录结构说明

* `signal_processor.py`: 信号预处理模块（陷波滤波、低通滤波）。
* `PlantMultimodalDataset.py`: 多模态数据集读取与异步信号对齐逻辑。
* `PlantTimeDomainEncoder.py`: 基于 1D CNN + Bi-GRU 的电压时序特征提取器。
* `PlantBiphasicActivation.py`: **[核心专利复现]** 双相生理激活层，提供快慢相解耦。
* `PlantFusionNet.py`: 多模态融合主网络。
* `main_train.py`: 自动化训练与生理指标监控总控脚本。

## 🛠️ 环境配置

本项目推荐使用 `uv` 或 `pip` 进行环境管理，要求 **Python 3.10+**。

### 1. 快速安装依赖

```bash
git clone https://github.com/mique11a/Multimodal-Plant-Emotion-Recognition.git
cd plant_condition_model
# 使用 uv 快速同步环境
uv pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121
# 或者使用标准 pip
pip install -r requirements.txt
```

### 2. 数据准备

请确保你的数据存放在 `dataset_filtered/` 目录下，并按以下结构组织：

```text
dataset_filtered/
├── normal/   # 存放正常状态的 CSV
├── touch/    # 存放触摸刺激的 CSV
└── light/    # 存放光照变化的 CSV

```

## 🚀 运行方法

执行总控脚本即可开始训练。脚本会自动检测 CUDA 加速，并实时打印各分类的 **$\alpha$ (快相占比)** 生理指标。

```bash
python main_train.py
```

## 📊 模块逻辑说明

* **输入对齐**：电压信号采样率为 250Hz，阻抗为单点采样。Dataset 模块通过均值填充和滑动窗口实现两者的特征对齐。
* **生理指标解读**：训练过程中输出的 $\alpha$ 值代表该状态是由“快速电信号（如动作电位）”还是“缓慢代谢（如变异电位）”引起的，数值越接近 1 代表响应越敏捷。


## 🚀 API 对接

**预测接口**

- 方法：`POST`
- 地址：`http://47.116.214.34:8000/predict`

**Request：**
```json
{
  "voltage": [0.1, 0.2, ...],
  "impedance": 7067.39
}
```
> `voltage`：250个浮点数（250Hz，采样1秒）  
> `impedance`：1个浮点数（原始阻抗值，单位与传感器一致）

**Response：**
```json
{
  "label": "touch",
  "label_id": 1,
  "confidence": 0.9231,
  "probabilities": {
    "normal": 0.0312,
    "touch": 0.9231,
    "light": 0.0301,
    "stress": 0.0156
  }
}
```
> `label` 四种取值：`normal`（正常）/ `touch`（触摸）/ `light`（光照）/ `stress`（胁迫）

**健康检查接口**

- 方法：`GET`
- 地址：`http://47.116.214.34:8000/health`
- Response：`{"status": "ok"}`

**API 在线文档**

- 地址：`http://47.116.214.34:8000/docs`

