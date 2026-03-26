import torch
import torch.nn as nn

class PlantTimeDomainEncoder(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=64):
        super(PlantTimeDomainEncoder, self).__init__()
        
        # ---------------------------------------------------------
        # Block 1: 宏观特征捕获器 (大卷积核 + 实例归一化)
        # ---------------------------------------------------------
        # kernel_size=31 相当于在 250Hz 下跨越了 124 毫秒的感受野，强迫网络忽略高频毛刺
        self.conv1 = nn.Conv1d(in_channels, 16, kernel_size=31, stride=2, padding=15)
        # 核心改动：加入 InstanceNorm1d。它会强制将每个样本片段的均值拉回 0。
        # 物理意义：无论植物当时的基线电压是 1.2V 还是 0.3V，网络只看“波动形状”，消除绝对漂移的干扰！
        self.in1 = nn.InstanceNorm1d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        # ---------------------------------------------------------
        # Block 2: 中等特征提取
        # ---------------------------------------------------------
        self.conv2 = nn.Conv1d(16, 32, kernel_size=11, stride=1, padding=5)
        self.in2 = nn.InstanceNorm1d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # 核心改动：加入 Dropout 防止网络过度依赖 CNN 提取到的强电压噪声
        self.dropout = nn.Dropout(p=0.3)
        
        # ---------------------------------------------------------
        # Block 3: 时序长程依赖捕获 (Bi-GRU)
        # ---------------------------------------------------------
        self.gru = nn.GRU(input_size=32, hidden_size=hidden_dim, 
                          num_layers=2, batch_first=True, bidirectional=True)
                          
    def forward(self, x):
        # x shape: [Batch, Channel=1, SeqLen=250]
        
        # CNN 阶段
        x = self.conv1(x)
        x = self.in1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.in2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.dropout(x)
        
        # 维度转换以适配 GRU: [Batch, Channel, SeqLen] -> [Batch, SeqLen, Channel]
        x = x.permute(0, 2, 1)
        
        # GRU 阶段
        out, h_n = self.gru(x)
        
        # 提取双向 GRU 的最后隐层状态
        # h_n shape: [num_layers * num_directions, Batch, hidden_dim]
        # 拼接正向和反向的最终输出，得到 128 维特征
        final_feat = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1) 
        
        return final_feat