import torch, time, sys
sys.path.insert(0, '.')
from PlantFusionNet import PlantFusionNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('设备:', device)

model = PlantFusionNet(num_classes=4).to(device)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
print('参数量:', total_params)

volt = torch.randn(1, 1, 250).to(device)
imp  = torch.randn(1, 1).to(device)

with torch.inference_mode():
    for _ in range(20):
        model(volt, imp)

N = 1000
t0 = time.perf_counter()
with torch.inference_mode():
    for _ in range(N):
        model(volt, imp)
t1 = time.perf_counter()
single_ms = (t1 - t0) / N * 1000
print(f'单样本推理延迟: {single_ms:.3f} ms')
print(f'理论最大吞吐: {1000/single_ms:.0f} 次/秒')

for bs in [1, 8, 32]:
    vb = torch.randn(bs, 1, 250).to(device)
    ib = torch.randn(bs, 1).to(device)
    with torch.inference_mode():
        for _ in range(10): model(vb, ib)
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(200): model(vb, ib)
    t1 = time.perf_counter()
    ms = (t1-t0)/200*1000
    print(f'batch={bs}: {ms:.2f} ms/batch, {bs/(ms/1000):.0f} samples/s')
