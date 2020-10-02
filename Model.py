from InputEmbedding import *
from EncoderDecoder import *
from MultiHeadedAttention import *
from PositionalEncoding import *
from FeedForward import *
import time


def make_model(src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1):
    c = copy.deepcopy  # 把copy.deepcopy命名为c，这样使下面的代码简洁一点。
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(  # 构造EncoderDecoder对象。它需要5个参数：Encoder、Decoder、src-embed、tgt-embed和Generator。
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),  # Decoder由N个DecoderLayer组成
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),  # src-embed是一个Embeddings层和一个位置编码层c(position)
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),  # tgt-embed也是类似的。
        Generator(d_model, tgt_vocab))  # 作用是把模型的隐单元变成输出词的概率

    # 随机初始化参数，这非常重要
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)  # Xavier初始化
    return model


class Batch:
    def __init__(self, src, trg=None, pad=0):  # Batch构造函数的输入是src和trg
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "创建Mask，使得我们不能attend to未来的词"
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


# 它遍历一个epoch的数据，然后调用forward，接着用loss_compute函数计算梯度，更新参数并且返回loss。
def run_epoch(data_iter, model, loss_compute):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        # loss_compute是一个函数，它的输入是模型的预测out，真实的标签序列batch.trg_y和batch的词个数。
        # 本来计算损失和更新参数比较简单，但是这里为了实现多GPU的训练，这个类就比较复杂了。
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                (i, loss / batch.ntokens, tokens / elapsed))
        start = time.time()
        tokens = 0
    return total_loss / total_tokens
