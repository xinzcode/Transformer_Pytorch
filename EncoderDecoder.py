import torch.nn.functional as F
import torch
import torch.nn as nn
import copy
import numpy as np


class EncoderDecoder(nn.Module):  # 继承Module类
    """
    标准的Encoder-Decoder架构。这是很多模型的基础
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super().__init__()  # 调用父类(超类)的一个方法(init)。
        # encoder和decoder都是构造的时候传入的，这样会非常灵活
        self.encoder = encoder
        self.decoder = decoder
        # 源语言和目标语言的embedding方法 ，参数为src或tgt
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        # generator后面会讲到，就是根据Decoder的隐状态输出当前时刻的词
        # 基本的实现就是隐状态输入一个全连接层，全连接层的输出大小是词的个数
        # 然后接一个softmax变成概率。
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):  # Module中定义了__call__()函数，该函数调用了forward()函数，类传入参数时会自动调用
        # 首先调用encode方法对输入进行编码，然后调用decode方法解码
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask)

    def encode(self, src, src_mask):
        # 调用encoder来进行编码，传入的参数embedding的src和src_mask
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):  # 即为encode获取的信息
        # 调用decoder
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


class Generator(nn.Module):
    # 根据Decoder的隐状态输出一个词，Decoder的后面两步（linear+softmax)
    # d_model是Decoder输出的大小，vocab是词典大小
    def __init__(self, d_model, vocab):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab)  # 全连接层进行线性变换

    # 全连接再加上一个softmax
    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)  # 按照指定维度在softmax基础上再log


# m = nn.LogSoftmax(dim=1)
# criterion = nn.NLLLoss()
# x = torch.randn(1, 5)
# y = torch.empty(1, dtype=torch.long).random_(5)
# loss = criterion(m(x), y)
# print(loss)


def clones(module, N):
    # 克隆N个完全相同的SubLayer，使用了copy.deepcopy
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
# 但是nn.ModuleList并不是Module(的子类)，因此它没有forward等方法，我们通常把它放到某个Module里。


class Encoder(nn.Module):
    "Encoder就是N个SubLayer的stack，最后加上一个LayerNorm。"
    # 不应该说有多层Encoder,应该说Encoder有多个SubLayer层
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        # layer是一个SubLayer，我们clone N个
        self.layers = clones(layer, N)
        # 再加一个LayerNorm层
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "逐层进行处理"
        for layer in self.layers:
            x = layer(x, mask)
        # 最后进行LayerNorm，后面会解释为什么最后还有一个LayerNorm。
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        # layer是一个SubLayer，我们clone N个
        self.layers = clones(layer, N)
        # 再加一个LayerNorm层
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        "逐层进行处理"
        for layer in self.layers:
            x = layer(x, mask)
        # 最后进行LayerNorm，后面会解释为什么最后还有一个LayerNorm。
        return self.norm(x)


# 若特征间具有不同的值范围时，因此梯度更新时，会来回震荡，经过较长的时间才能达到局部最优值或全局最优值。
# 为了解决该模型问题，我们需要归一化数据，我们确保不同的特征具有相同的值范围，这样梯度下降可以很快的收敛。
class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# 不管是Self-Attention还是全连接层，都首先是LayerNorm，然后是Self-Attention/Dense，然后是Dropout，最后是残差连接。
# 构造norm+dropout+add，这里面有很多可以重用的代码，我们把它封装成SublayerConnection。
class SublayerConnection(nn.Module):
    """
    LayerNorm + sublayer(Self-Attenion/Dense) + dropout + 残差连接
    为了简单，把LayerNorm放到了前面，这和原始论文稍有不同，原始论文LayerNorm在最后。
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):  # 这个方法需要两个参数，一个是输入Tensor，一个是一个callable，并且这个callable可以用一个参数来调用
        "sublayer是传入的参数，参考DecoderLayer，它可以当成函数调用，这个函数的有一个输入参数"
        return x + self.dropout(sublayer(self.norm(x)))


# 构造Self-Attention或者Dense
class EncoderLayer(nn.Module):
    "EncoderLayer由self-attn和feed forward组成"
    # 为了复用，这里的self_attn层和feed_forward层也是传入的参数，这里只构造两个SublayerConnection。
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn  # self_attn函数需要4个参数(Query的输入,Key的输入,Value的输入和Mask)
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)  # 自注意层和前向层都需要进行norm+dropout+add
        self.size = size

    # def forward(self, x, mask):
    #     "Follow Figure 1 (left) for connections."
    #     x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))  # 使用lambda的技巧把它变成一个参数x的函数(mask可以看成已知的数)。
    #     return self.sublayer[1](x, self.feed_forward)

    # 解释：
    # self_attn有4个参数，但是我们知道在Encoder里，前三个参数都是输入y，第四个参数是mask。
    # 这里mask是已知的，因此我们可以用lambda的技巧它变成一个参数的函数z = lambda y: self.self_attn(y, y, y, mask)，这个函数的输入是y。
    def forward(self, x, mask):
        z = lambda y: self.self_attn(y, y, y, mask)
        x = self.sublayer[0](x, z)  # z就等于sublayer中forward 的 sublayer
        # self.sublayer[0]是个callable，self.sublayer[0] (x, z)会调用self.sublayer[0].call(x, z)，
        # 然后会调用SublayerConnection.forward(x, z)，然后会调用sublayer(self.norm(x))，sublayer就是传入的参数z，因此就是z(self.norm(x))。
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    "Decoder包括self-attn, src-attn, 和feed forward "

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn  # 比EncoderLayer多了一个src-attn层。
        # 这是Decoder时attend to Encoder的输出(memory)。src-attn和self-attn的实现是一样的，只不过使用的Query，Key和Value的输入不同。
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):  # 多一个来自Encoder的memory
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))  # self-attention
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))  # encoder-decoder attention
        return self.sublayer[2](x, self.feed_forward)


# Decoder和Encoder有一个关键的不同：Decoder在解码第t个时刻的时候只能使用1…t时刻的输入，
# 而不能使用t+1时刻及其之后的输入。因此我们需要一个函数来产生一个Mask矩阵，
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)  # 全初始化为1
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')  # 返回函数的上三角矩阵(其他位为0），k=1表示对角线的位置上移1个对角线，k默认是0
    return torch.from_numpy(subsequent_mask) == 0  # torch.from_numpy()方法把数组转换成张量，且二者共享内存。matrix == 0来实现把0变成1，把1变成0。

# print(subsequent_mask(5))
# 输出：
#   1  0  0  0  0
#   1  1  0  0  0
#   1  1  1  0  0
#   1  1  1  1  0
#   1  1  1  1  1







