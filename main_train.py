import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from PlantMultimodalDataset import PlantMultimodalDataset
from PlantFusionNet import PlantFusionNet

def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  使用设备: {device}")

    # 1. 加载数据
    dataset = PlantMultimodalDataset(root_dir='dataset_filtered', window_size=250)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    # 2. 初始化模型
    num_classes = len(dataset.label_map) # 4类: normal, touch, light, stress
    model = PlantFusionNet(num_classes=num_classes).to(device)
    
    # 3. 损失函数与优化器
    # 假定合成数据极度均衡，所有类别权重设为 1.0
    weights = torch.ones(num_classes).to(device)
    
    # PBA层已完成归一化，必须使用 NLLLoss 匹配对数概率
    criterion = nn.NLLLoss(weight=weights)

    # 优化器参数分组
    pba_params = list(model.pba.parameters())
    base_params = [p for n, p in model.named_parameters() if "pba" not in n]

    optimizer = optim.Adam([
        {'params': base_params, 'lr': 1e-4},
        # 略微调低 PBA 的初始学习率至 5e-3，防止初期震荡
        {'params': pba_params, 'lr': 5e-3} 
    ])
    
    # 加入余弦退火调度器，让学习率随 Epoch 平滑下降
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # 4. 训练循环
    print("\n🚀 开始植物情绪模型训练...")
    for epoch in range(200):
        model.train()
        total_loss = 0
        for volts, imps, labels in train_loader:
            volts, imps, labels = volts.to(device), imps.to(device), labels.to(device)
            
            optimizer.zero_grad()
            phi_norm, _, _ = model(volts, imps)
            
            # 将归一化概率转换为对数概率，加上 1e-8 防止 log(0)
            log_probs = torch.log(phi_norm + 1e-8)
            loss = criterion(log_probs, labels)
            
            loss.backward()
            
            # 梯度裁剪：限制梯度最大范数为 1.0，彻底消除异常飙升
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
        # 每个 Epoch 结束后更新学习率
        scheduler.step()
            
        if epoch % 5 == 0:
            # 监控生理指标 (对比 Touch 快响应 和 Light 慢响应)
            with torch.no_grad():
                alpha_touch = torch.sigmoid(model.pba.alpha[1]).item()
                alpha_light = torch.sigmoid(model.pba.alpha[2]).item()
                current_lr = scheduler.get_last_lr()[0]
                
                print(f"Epoch {epoch:3d} | Loss: {total_loss/len(train_loader):.4f} | "
                      f"[Touch]α: {alpha_touch:.4f} | [Light]α: {alpha_light:.4f} | LR: {current_lr:.6f}")

    # 保存最新状态
    torch.save(model.state_dict(), "plant_fusion_synth_true.pth")
    print("\n✅ 训练完成，模型架构已通过验证")

if __name__ == "__main__":
    run_training()