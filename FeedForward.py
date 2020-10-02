import torch.nn as nn
import torch.nn.functional as F


# 全连接层,以独立并行计算，由两个线性变换以及它们之间的ReLU激活组成
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):  # 输入输出是d_model维，隐单元个数为d_ff
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))  # 在两个线性变换之间除了ReLu还使用了一个Dropout。
