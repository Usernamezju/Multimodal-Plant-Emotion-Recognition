import torch
import torch.nn as nn
from PlantTimeDomainEncoder import PlantTimeDomainEncoder
from PlantBiphasicActivation import PlantBiphasicActivation

class PlantFusionNet(nn.Module):
    def __init__(self, num_classes=3):
        super(PlantFusionNet, self).__init__()
        
        # 1. 时序分支 (Bi-GRU 输出维度为 hidden_size * 2 = 128)
        self.volt_branch = PlantTimeDomainEncoder(
            input_channels=1, 
            hidden_size=64, 
            num_classes=128 # 这里的 num_classes 实际上是特征输出维度
        )
        
        # 2. 阻抗分支 (MLP)
        self.imp_branch = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 64), # 映射到 64 维
            nn.ReLU()
        )
        
        # 3. 融合投影层 (128[电位] + 64[阻抗] = 192)
        self.fusion_projection = nn.Linear(128 + 64, num_classes)
        
        # 4. 专利核心 PBA 层
        self.pba = PlantBiphasicActivation(num_classes=num_classes)

    def forward(self, volt, imp):
        # volt: [B, 1, 250], imp: [B, 1]
        
        # 提取时序特征: [B, 128]
        v_feat = self.volt_branch(volt) 
        
        # 提取阻抗特征: [B, 64]
        i_feat = self.imp_branch(imp)
        
        # 模态融合: [B, 192]
        combined = torch.cat((v_feat, i_feat), dim=1)
        
        # 映射到 Logits 空间
        logits = self.fusion_projection(combined)
        
        # PBA 生理判别
        phi_norm, f_fast, f_slow = self.pba(logits)
        
        return phi_norm, f_fast, f_slow