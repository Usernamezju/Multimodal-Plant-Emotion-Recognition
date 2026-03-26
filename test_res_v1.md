# model/plant_fusion_best_v1.pth

(plant) yuki_noa@LAPTOP-0O7OKH4A ~/plant_condition_model (main)> uv run test.py
⚙️ 正在加载模型权重: plant_fusion_best_real.pth ...
📂 正在加载真实数据集流水线...
📊 阻抗 Z-Score 统计完成: Mean=6018.39, Std=275.15
✅ 数据加载与切片完成，共提取到 4000 个有效样本。

🔍 正在抽取真实的normal样本进行推理...

🌿 模型推理完成，JSON 输出：

{
  "code": 200,
  "msg": "success",
  "data": {
    "timestamp": 1774538237471,
    "primary_emotion": "Touch",
    "confidence": 0.8938,
    "probabilities": {
      "Normal": 0.0701,
      "Touch": 0.8938,
      "Light": 0.0308,
      "Stress": 0.0052
    },
    "physiological_indicators": {
      "fast_response": 0.8714,
      "slow_response": 0.1286
    }
  }
}
(plant) yuki_noa@LAPTOP-0O7OKH4A ~/plant_condition_model (main)> uv run test.py
⚙️ 正在加载模型权重: plant_fusion_best_real.pth ...
📂 正在加载真实数据集流水线...
📊 阻抗 Z-Score 统计完成: Mean=6018.39, Std=275.15
✅ 数据加载与切片完成，共提取到 4000 个有效样本。

🔍 正在抽取真实的touch样本进行推理...

🌿 模型推理完成，JSON 输出：

{
  "code": 200,
  "msg": "success",
  "data": {
    "timestamp": 1774538249714,
    "primary_emotion": "Touch",
    "confidence": 0.4572,
    "probabilities": {
      "Normal": 0.191,
      "Touch": 0.4572,
      "Light": 0.3397,
      "Stress": 0.012
    },
    "physiological_indicators": {
      "fast_response": 0.8714,
      "slow_response": 0.1286
    }
  }
}
(plant) yuki_noa@LAPTOP-0O7OKH4A ~/plant_condition_model (main)> uv run test.py
⚙️ 正在加载模型权重: plant_fusion_best_real.pth ...
📂 正在加载真实数据集流水线...
📊 阻抗 Z-Score 统计完成: Mean=6018.39, Std=275.15
✅ 数据加载与切片完成，共提取到 4000 个有效样本。

🔍 正在抽取真实的light样本进行推理...

🌿 模型推理完成，JSON 输出：

{
  "code": 200,
  "msg": "success",
  "data": {
    "timestamp": 1774538268697,
    "primary_emotion": "Light",
    "confidence": 0.8411,
    "probabilities": {
      "Normal": 0.1519,
      "Touch": 0.0061,
      "Light": 0.8411,
      "Stress": 0.0008
    },
    "physiological_indicators": {
      "fast_response": 0.1625,
      "slow_response": 0.8375
    }
  }
}