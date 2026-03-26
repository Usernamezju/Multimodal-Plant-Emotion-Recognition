import torch
import time
import json
import warnings
from PlantFusionNet import PlantFusionNet
from PlantMultimodalDataset import PlantMultimodalDataset

warnings.filterwarnings("ignore", category=FutureWarning)

TYPE = "light"

def load_model(weights_path, device):
    print(f"⚙️ 正在加载模型权重: {weights_path} ...")
    model = PlantFusionNet(num_classes=4).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True), strict=False)
    model.eval() 
    return model

def get_real_sample_from_dataset(dataset, target_emotion="touch"):
    """直接从数据集中抓取一个真实的指定类别样本"""
    target_label = dataset.label_map[target_emotion]
    
    for idx in range(len(dataset)):
        volt, imp, label = dataset[idx]
        if label.item() == target_label:
            # 找到一个目标样本后直接返回 (增加批量维度)
            return volt.unsqueeze(0), imp.unsqueeze(0)
            
    raise ValueError(f"🚨 在数据集中没有找到 {target_emotion} 的样本！")

def predict_to_json(model, volt_tensor, imp_tensor, dataset, device):
    volt_tensor = volt_tensor.to(device)
    imp_tensor = imp_tensor.to(device)
    
    with torch.no_grad():
        phi_norm, f_fast, f_slow = model(volt_tensor, imp_tensor)
        
    probs = phi_norm.squeeze().cpu().tolist() 
    
    # 根据 dataset 的 label_map 动态生成反向映射表
    reverse_map = {v: k.capitalize() for k, v in dataset.label_map.items()}
    
    confidence = max(probs)
    pred_idx = probs.index(confidence)
    primary_emotion = reverse_map[pred_idx]
    
    # 提取所有类别的 alpha 权重
    alpha_weights = torch.sigmoid(model.pba.alpha).cpu().tolist()
    
    # 强制打印判定类别的真实生理指标
    current_alpha = alpha_weights[pred_idx]
    
    api_response = {
        "code": 200,
        "msg": "success",
        "data": {
            "timestamp": int(time.time() * 1000),
            "primary_emotion": primary_emotion,
            "confidence": round(confidence, 4),
            "probabilities": {
                reverse_map[0]: round(probs[0], 4),
                reverse_map[1]: round(probs[1], 4),
                reverse_map[2]: round(probs[2], 4),
                reverse_map[3]: round(probs[3], 4)
            },
            "physiological_indicators": {
                "fast_response": round(current_alpha, 4),          
                "slow_response": round(1.0 - current_alpha, 4)     
            }
        }
    }
    
    return json.dumps(api_response, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    WEIGHTS_FILE = "model/plant_fusion_best_v1.pth"
    DATA_ROOT = "dataset_synthetic_filtered" # 确保这是你真实数据的目录
    
    # 1. 初始化模型
    plant_model = load_model(WEIGHTS_FILE, DEVICE)
    
    # 2. 初始化数据集 (利用它原生的读取和预处理管道)
    print("📂 正在加载真实数据集流水线...")
    dataset = PlantMultimodalDataset(root_dir=DATA_ROOT, window_size=250)
    
    # 3. 抽取真实的 Touch 样本
    print("\n🔍 正在抽取真实的" + TYPE + "样本进行推理...")
    v_tensor, i_tensor = get_real_sample_from_dataset(dataset, target_emotion=TYPE)
    
    # 4. 执行推理
    final_json = predict_to_json(plant_model, v_tensor, i_tensor, dataset, DEVICE)
    print("\n🌿 模型推理完成，JSON 输出：\n")
    print(final_json)