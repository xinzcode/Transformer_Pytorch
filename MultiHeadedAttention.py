import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from EncoderDecoder import clones


# 假设Q是(30,8,33,64)，其中30是batch，8是head个数，33是序列长度，64是每个时刻的特征数。K和Q的shape必须相同的，而V可以不同，但是这里的实现shape也是相同的。
def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
        / math.sqrt(d_k)
    # 和公式里稍微不同的是，这里的Q和K都是4d的Tensor，包括batch和head维度。
    # matmul会把query和key的最后两维进行矩阵乘法，这样效率更高。
    # 输出的是(30, 8, 33, 33)，前面两维不看，那么是一个(33, 33)的attention矩阵a，
    # aij表示时刻i(batch_num) attend to j(head_num)的得分(还没有经过softmax变成概率)。
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
        # 用于把mask是0的变成一个很小的数，这样后面经过softmax之后的概率就很接近零(但是理论上还是用了很少一点点未来的信息)。
        # 这里mask是(30, 1, 1, 33)的tensor，因为8个head的mask都是一样的
        # 因为8个head的mask都是一样的，所有第二维是1
        # 这里是self-attention的mask，所以每个时刻都可以attend到所有其它时刻，所有第三维也是1，
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)  # 原始论文没有的
    return torch.matmul(p_attn, value), p_attn
    # p_attn是(30, 8, 33, 33)，value是(30, 8, 33, 64)，我们只看后两维，(33x33) x (33x64)最终得到33x64。


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h  # 除以Head个数
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)  # 构造4个(d_model x d_model)的矩阵，
        # 前3个用于对query，key和value进行变换，而最后一个对8个head拼接后的向量再做一次变换。
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)  # 最后是构造一个Dropout层。

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # 所有h个head的mask都是相同的
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) 首先使用线性变换，然后把d_model分配给h个Head，每个head为d_k=d_model/h
        # 根据输入query，key和value计算变换后的Multi-Head的query，key和value
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) 使用attention函数计算
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) 把8个head的64维向量拼接成一个512的向量。然后再使用一个线性变换(512,521)，shape不变。
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)  # 最后使用一个线性变换进行降维。


