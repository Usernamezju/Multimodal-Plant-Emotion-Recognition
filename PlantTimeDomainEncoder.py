import torch.nn as nn

class PlantTimeDomainEncoder(nn.Module):
    def __init__(self, input_channels=1, hidden_size=64, num_classes=5):
        """
        专门处理植物电位信号的时序分支网络
        :param input_channels: 输入维度，通常为1（电压）
        :param hidden_size: GRU隐藏层维度
        :param num_classes: 最终分类数量（对应PBA的输入）
        """
        super(PlantTimeDomainEncoder, self).__init__()
        
        # 1. 1D CNN 分支：捕捉局部波形特征（如触摸尖峰）
        self.conv_block = nn.Sequential(
            # 第一层卷积，卷积核大小为7，增加感受野
            nn.Conv1d(input_channels, 32, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2), # 降采样
            
            # 第二层卷积，提取深层抽象特征
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32) # 统一输出长度，方便后续接时序模型
        )
        
        # 2. GRU 分支：记忆长期生理状态（如光照漂移）
        # batch_first=True 意味着输入形状为 (batch, seq_len, features)
        self.gru = nn.GRU(input_size=64, hidden_size=hidden_size, 
                          num_layers=2, batch_first=True, bidirectional=True)
        
        # 3. 特征投影层：将GRU的隐藏状态映射到PBA的输入空间
        # 因为是双向GRU，所以维度要乘以2
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        """
        :param x: 原始电压序列，形状 (batch_size, 1, seq_len)
        """
        # --- CNN 局部特征提取 ---
        # 输入 (batch, 1, seq_len) -> 输出 (batch, 64, 32)
        x = self.conv_block(x)
        
        # --- 维度调整，适配 GRU ---
        # GRU需要 (batch, seq_len, features)，所以要置换一下维度
        x = x.permute(0, 2, 1) 
        
        # --- GRU 序列建模 ---
        # out 形状: (batch, seq_len, hidden_size*2)
        out, _ = self.gru(x)
        
        # 我们取序列的最后一个时间步的特征作为全局表示
        last_time_step = out[:, -1, :]
        
        # --- 投影到 logit 空间 ---
        logits = self.fc(last_time_step)
        
        return logits