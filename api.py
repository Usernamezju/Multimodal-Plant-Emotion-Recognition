from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from PlantFusionNet import PlantFusionNet

app = FastAPI()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PlantFusionNet(num_classes=4).to(device)
model.load_state_dict(torch.load("model/plant_fusion_best_v1.pth", map_location=device, weights_only=True))
model.eval()

IMP_MEAN = 7067.39
IMP_STD  = 322.02

LABEL_NAMES = ["normal", "touch", "light", "stress"]

class PredictRequest(BaseModel):
    voltage: list[float]
    impedance: float

class PredictResponse(BaseModel):
    label: str
    label_id: int
    confidence: float
    probabilities: dict

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if len(req.voltage) != 250:
        raise HTTPException(status_code=400, detail=f"voltage 必须是250个点，收到{len(req.voltage)}个")
    
    try:
        volt = torch.tensor(req.voltage, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        norm_imp = (req.impedance - IMP_MEAN) / IMP_STD
        imp = torch.tensor([[norm_imp]], dtype=torch.float32).to(device)

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
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ok"}
