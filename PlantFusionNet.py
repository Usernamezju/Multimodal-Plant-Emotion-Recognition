import torch
import torch.nn as nn
import torch.nn.functional as F
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
        v_feat = self.volt_branch(volt) 
        i_feat = self.imp_branch(imp)
        
        combined = torch.cat((v_feat, i_feat), dim=1)
        
        # 【修改点】：加入 Dropout 逼迫多模态均衡发展，防止模型只依赖阻抗偷懒
        combined = F.dropout(combined, p=0.3, training=self.training)
        
        logits = self.fusion_projection(combined)
        phi_norm, f_fast, f_slow = self.pba(logits)
        return phi_norm, f_fast, f_slow