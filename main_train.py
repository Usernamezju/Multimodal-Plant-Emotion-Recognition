import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from PlantMultimodalDataset import PlantMultimodalDataset
from PlantFusionNet import PlantFusionNet
import os

def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  使用设备: {device}")

    # 1. 加载数据
    dataset = PlantMultimodalDataset(root_dir='dataset_real_condition_filtered', window_size=250)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False) # 验证集不需要打乱

    # 2. 初始化模型
    num_classes = len(dataset.label_map)
    model = PlantFusionNet(num_classes=num_classes).to(device)

    # 加载预训练参数（从真实数据训练的检查点继续）
    checkpoint_path = "plant_fusion_train_by_true_data.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    print(f"✅ 已加载预训练参数: {checkpoint_path}")
    
    # 3. 损失函数与优化器
    criterion = nn.NLLLoss()

    pba_params = list(model.pba.parameters())
    base_params = [p for n, p in model.named_parameters() if "pba" not in n]

    optimizer = optim.Adam([
        {'params': base_params, 'lr': 1e-4},
        {'params': pba_params, 'lr': 5e-3} 
    ])
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    # 🔥 新增：用于记录最佳验证损失，实现“早停”和“最优保存”
    best_val_loss = float('inf')
    best_model_path = "plant_fusion_best_real.pth"

    # 4. 训练与验证循环
    print("\n🚀 开始植物情绪模型训练 (包含全量验证逻辑)...")
    for epoch in range(200):
        # ------------------ [训练阶段] ------------------
        model.train()
        train_loss = 0.0
        for volts, imps, labels in train_loader:
            volts, imps, labels = volts.to(device), imps.to(device), labels.to(device)
            
            optimizer.zero_grad()
            phi_norm, _, _ = model(volts, imps)
            
            log_probs = torch.log(phi_norm + 1e-8)
            loss = criterion(log_probs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        scheduler.step()

        # ------------------ [验证阶段] ------------------
        model.eval() # 开启评估模式 (关闭 Dropout 和 BatchNorm 的动态更新)
        val_loss = 0.0
        with torch.no_grad(): # 验证阶段不需要计算梯度，节省显存并加速
            for v_volts, v_imps, v_labels in val_loader:
                v_volts, v_imps, v_labels = v_volts.to(device), v_imps.to(device), v_labels.to(device)
                
                v_phi_norm, _, _ = model(v_volts, v_imps)
                v_log_probs = torch.log(v_phi_norm + 1e-8)
                v_loss = criterion(v_log_probs, v_labels)
                val_loss += v_loss.item()
                
        avg_val_loss = val_loss / len(val_loader)

        # ------------------ [指标监控与模型保存] ------------------
        # 如果当前验证损失是历史最低，则保存权重
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            saved_marker = "⭐" # 打个星号标记
        else:
            saved_marker = "  "

        if epoch % 5 == 0:
            with torch.no_grad():
                alpha_touch = torch.sigmoid(model.pba.alpha[dataset.label_map["touch"]]).item()
                alpha_light = torch.sigmoid(model.pba.alpha[dataset.label_map["light"]]).item()
                current_lr = scheduler.get_last_lr()[0]
                
                print(f"Epoch {epoch:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} {saved_marker} | "
                      f"[Touch]α: {alpha_touch:.4f} | [Light]α: {alpha_light:.4f}")

    print(f"\n✅ 训练结束！最佳验证损失为 {best_val_loss:.4f}，模型已保存至 {best_model_path}")
    torch.save(model.state_dict(), best_model_path + "_last")

if __name__ == "__main__":
    run_training()