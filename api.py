from fastapi import FastAPI
from pydantic import BaseModel
import torch
from PlantFusionNet import PlantFusionNet

app = FastAPI()

# 启动时加载模型（只加载一次）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PlantFusionNet(num_classes=4).to(device)
model.load_state_dict(torch.load("plant_fusion_train_by_true_data.pth", map_location=device, weights_only=True))
model.eval()

LABEL_NAMES = ["normal", "touch", "light", "stress"]

# ── Request 结构 ──
class PredictRequest(BaseModel):
    voltage: list[float]   # 250个电压采样点（250Hz，1秒）
    impedance: float       # 单点阻抗值

# ── Response 结构 ──
class PredictResponse(BaseModel):
    label: str             # 预测类别名称
    label_id: int          # 类别编号 0-3
    confidence: float      # 置信度 0.0~1.0
    probabilities: dict    # 各类别概率

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    volt = torch.tensor(req.voltage, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    imp  = torch.tensor([[req.impedance]], dtype=torch.float32).to(device)

    with torch.inference_mode():
        phi_norm, _, _ = model(volt, imp)
        probs = torch.softmax(phi_norm, dim=1)[0]
        pred_id = probs.argmax().item()

    return PredictResponse(
        label=LABEL_NAMES[pred_id],
        label_id=pred_id,
        confidence=round(probs[pred_id].item(), 4),
        probabilities={name: round(probs[i].item(), 4) for i, name in enumerate(LABEL_NAMES)}
    )

@app.get("/health")
def health():
    return {"status": "ok"}
