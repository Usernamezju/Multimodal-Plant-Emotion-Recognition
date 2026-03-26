import torch
import torch.nn as nn
import torch.nn.functional as F
from PlantTimeDomainEncoder import PlantTimeDomainEncoder
from PlantBiphasicActivation import PlantBiphasicActivation

class PlantFusionNet(nn.Module):
    def __init__(self, num_classes=4): # 建议默认设为4，匹配实际类别数
        super(PlantFusionNet, self).__init__()
        
        # 1. 时序分支 (Bi-GRU 输出维度为 hidden_size * 2 = 128)
        # 【修改点】：移除了不再需要的 num_classes
        self.volt_branch = PlantTimeDomainEncoder(
            in_channels=1, 
            hidden_dim=64
        )
        
        # 2. 阻抗分支 (MLP)
        self.imp_branch = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, 64), # 映射到 64 维
            nn.ReLU()
        )
        
        # 3. 模态对齐层
        # 【新增点】：用于在拼接后立刻对齐电位和阻抗的数值分布
        self.layer_norm = nn.LayerNorm(128 + 64)
        
        # 4. 融合投影层 (128[电位] + 64[阻抗] = 192)
        self.fusion_projection = nn.Linear(128 + 64, num_classes)
        
        # 5. 专利核心 PBA 层
        self.pba = PlantBiphasicActivation(num_classes=num_classes)

    def forward(self, volt, imp):
        # 提取双模态独立特征
        v_feat = self.volt_branch(volt) 
        i_feat = self.imp_branch(imp)
        
        # 特征拼接 [Batch, 192]
        combined = torch.cat((v_feat, i_feat), dim=1)
        
        # 【新增点】：跨模态归一化，消除 Scale 差异带来的梯度吞噬
        combined = self.layer_norm(combined)
        
        # 引入随机失活，逼迫网络均衡地从两个模态汲取信息
        combined = F.dropout(combined, p=0.3, training=self.training)
        
        # 降维投影与双相激活
        logits = self.fusion_projection(combined)
        phi_norm, f_fast, f_slow = self.pba(logits)
        
        return phi_norm, f_fast, f_slow