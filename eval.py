import torch
import os
from torch.utils.data import DataLoader, random_split
from PlantMultimodalDataset import PlantMultimodalDataset
from PlantFusionNet import PlantFusionNet

LABEL_NAMES = ["normal", "touch", "light", "stress"]

CHECKPOINTS = [
    "model/plant_fusion_best_v1.pth",
    "model/plant_fusion_last_v1.pth",
]




def eval_one(model, val_loader, device, num_classes):
    """对一个已加载好的模型跑推理，返回整体准确率和各类别准确率"""
    correct = 0
    total = 0
    class_correct = [0] * num_classes
    class_total   = [0] * num_classes

    with torch.inference_mode():
        for volts, imps, labels in val_loader:
            volts  = volts.to(device)
            imps   = imps.to(device)
            labels = labels.to(device)

            phi_norm, _, _ = model(volts, imps)
            preds = phi_norm.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total   += labels.size(0)

            for pred, label in zip(preds, labels):
                class_total[label.item()]   += 1
                class_correct[label.item()] += int(pred == label)

    overall = correct / total * 100 if total > 0 else 0.0
    per_class = [
        class_correct[i] / class_total[i] * 100 if class_total[i] > 0 else float('nan')
        for i in range(num_classes)
    ]
    return overall, per_class, class_correct, class_total


def run_eval(dataset_dir="dataset_real_condition_filtered", batch_size=32, val_ratio=0.2):
    #更改数据集
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}\n")

    # 1. 加载数据集（固定随机种子保证所有模型用同一验证集）
    dataset = PlantMultimodalDataset(root_dir=dataset_dir, window_size=250)
    num_classes = len(dataset.label_map)

    val_size   = int(val_ratio * len(dataset))
    train_size = len(dataset) - val_size
    _, val_ds  = random_split(dataset, [train_size, val_size],
                              generator=torch.Generator().manual_seed(42))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    print(f"验证集大小: {len(val_ds)} 个样本\n")

    # 2. 逐个评估每个 checkpoint
    results = []
    for ckpt in CHECKPOINTS:
        if not os.path.exists(ckpt):
            print(f"[跳过] 找不到文件: {ckpt}\n")
            continue

        model = PlantFusionNet(num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.eval()

        overall, per_class, class_correct, class_total = eval_one(
            model, val_loader, device, num_classes
        )
        results.append((ckpt, overall, per_class, class_correct, class_total, model))
        print(f"完成: {ckpt}")

    # 3. 汇总对比表
    col = 22
    header = f"{'模型':<{col}}" + "".join(f"{n:>10}" for n in LABEL_NAMES) + f"{'整体':>10}"
    sep = "-" * len(header)

    print(f"\n{'='*len(header)}")
    print("  各模型准确率对比 (%)")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)

    best_overall = -1
    best_name = ""
    for ckpt, overall, per_class, _, _, _ in results:
        short = os.path.basename(ckpt).replace(".pth", "")
        # 截断过长的名字
        short = short[:col-1] if len(short) >= col else short
        row = f"{short:<{col}}"
        for acc in per_class:
            row += f"{acc:>9.1f}%"
        row += f"{overall:>9.1f}%"
        print(row)
        if overall > best_overall:
            best_overall = overall
            best_name = ckpt

    print(sep)
    print(f"\n最佳模型: {os.path.basename(best_name)}  ({best_overall:.1f}%)\n")

    # 4. PBA 生理参数（仅最佳模型）
    best_model = next(m for ckpt, _, _, _, _, m in results if ckpt == best_name)
    print(f"PBA 生理参数 [{os.path.basename(best_name)}]  (alpha > 0.5 = 快响应AP主导):")
    with torch.inference_mode():
        for i, name in enumerate(LABEL_NAMES):
            alpha = torch.sigmoid(best_model.pba.alpha[i]).item()
            tag = "快响应(AP)主导" if alpha > 0.5 else "慢响应(VP)主导"
            print(f"  {name:<8}: alpha={alpha:.4f}  {tag}")
    print()


if __name__ == "__main__":
    run_eval()
