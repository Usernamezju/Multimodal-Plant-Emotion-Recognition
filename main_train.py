import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from PlantMultimodalDataset import PlantMultimodalDataset
from PlantFusionNet import PlantFusionNet

def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  使用设备: {device}")

    # 1. 加载数据并统计归一化参数
    dataset = PlantMultimodalDataset(root_dir='dataset_filtered', window_size=250)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # 随机分配训练组和检验组
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # 2. 初始化模型
    num_classes = len(dataset.label_map) # 4类: normal, touch, light, stress
    model = PlantFusionNet(num_classes=num_classes).to(device)
    
    # 3. 损失函数与优化器 (针对数据缺失进行类别加权)
    weights = torch.tensor([1.0, 3.0, 3.0, 0.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    # 为 PBA 参数设置更高的学习率
    pba_params = list(model.pba.parameters())
    base_params = [p for n, p in model.named_parameters() if "pba" not in n]
    optimizer = optim.Adam([
        {'params': base_params, 'lr': 1e-4},
        {'params': pba_params, 'lr': 1e-2} # 给生理指标参数 10 倍的学习率
    ])
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 4. 训练循环
    print("\n开始植物情绪模型训练...")
    for epoch in range(200):
        model.train()
        total_loss = 0
        for volts, imps, labels in train_loader:
            volts, imps, labels = volts.to(device), imps.to(device), labels.to(device)
            
            optimizer.zero_grad()
            phi_norm, _, _ = model(volts, imps)
            loss = criterion(phi_norm, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if epoch % 5 == 0:
            # 打印生理指标监控 (以 Touch 类别为例)
            # print(f"DEBUG: Logits range: {torch.logit.min().item():.2f} to {torch.logit.max().item():.2f}")
            with torch.no_grad():
                alpha_touch = torch.sigmoid(model.pba.alpha[1]).item()
                print(f"Epoch {epoch} | Loss: {total_loss/len(train_loader):.4f} | [Touch]快相占比α: {alpha_touch:.4f}")

    # 5. 保存模型
    torch.save(model.state_dict(), "plant_fusion_model.pth")
    print("\n✅ 训练完成，模型已保存。")

if __name__ == "__main__":
    run_training()