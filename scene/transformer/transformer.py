import torch.nn as nn
import torch
import torch.nn.functional as F

from scene.transformer.layer_norm import LayerNorm
from scene.transformer.multi_head_attention import MultiHeadAttention
from scene.transformer.position_wise_feed_forward import PositionwiseFeedForward

class CrossModalAttention(nn.Module):
    def __init__(self, feature_dim, num_heads):
        super(CrossModalAttention, self).__init__()
        self.query_linear = nn.Linear(64, feature_dim)
        self.key_linear = nn.Linear(feature_dim, feature_dim)
        self.value_linear = nn.Linear(feature_dim, feature_dim)
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = (feature_dim // num_heads) ** 0.5

        self.offset_linear = nn.Linear(feature_dim, feature_dim)
        self.time_bias = nn.Parameter(torch.tensor(1.0))  # 初始时间偏置为1，可学习

    def forward(self, x, enc_source):
        # 生成查询、键、和值
        Q = self.query_linear(x)  # 音频特征生成查询
        K = self.key_linear(enc_source)  # 面部和视角特征生成键
        V = self.value_linear(enc_source)  # 面部和视角特征生成值

        # 计算注意力分数
        Q = Q.view(Q.size(0), Q.size(1), self.num_heads, self.head_dim).transpose(1, 2)  # 转换成多头格式
        K = K.view(K.size(0), K.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(V.size(0), V.size(1), self.num_heads, self.head_dim).transpose(1, 2)

        # 1. 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [B, n_heads, Q_len, K_len]

        # 2. 可选增强：对 time token 添加偏置
        time_index = 4  # 例如第5个token是time_token（你在enc_source里拼接的顺序必须对应）
        bias = torch.zeros_like(attention_scores)
        bias[:, :, :, time_index] += self.time_bias  # self.time_bias 是可学习参数
        attention_scores = attention_scores + bias

        # 3. 正常 softmax
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 加权和值
        out = torch.matmul(attention_weights, V)
        out = out.transpose(1, 2).contiguous().view(out.size(0), Q.size(2), -1)

        x_ = self.offset_linear(out)
        x = x + x_

        return x, attention_weights


class CrossModalAttentionModule(nn.Module):
    """
    交叉模态注意力模块，用于初始化模型
    """

    def __init__(self, args):
        super(CrossModalAttentionModule, self).__init__()
        self.cross_modal_attention = CrossModalAttention(feature_dim=args.d_model, num_heads=args.n_head)

    def forward(self, x, enc_source):
        """
        调用 CrossModalAttention 实现模态融合
        """
        x, attention_weights = self.cross_modal_attention(x, enc_source)
        return x, attention_weights


class MultiModalFusion(nn.Module):
    def __init__(self, feature_dim):
        super(MultiModalFusion, self).__init__()
        self.fc = nn.Linear(feature_dim, feature_dim)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, enc_source):
        enc_source = self.norm(F.relu(self.fc(enc_source)))
        return enc_source


# class CrossModalAttention(nn.Module):
#     def __init__(self, feature_dim):
#         super(CrossModalAttention, self).__init__()
#         self.query_linear = nn.Linear(feature_dim, feature_dim)
#         self.key_linear = nn.Linear(feature_dim, feature_dim)
#         self.value_linear = nn.Linear(feature_dim, feature_dim)
#         self.feature_dim = feature_dim
#         self.scale = feature_dim ** 0.5  # 这里不再除以head_dim
#
#         self.offset_linear = nn.Linear(feature_dim, feature_dim)
#
#     def forward(self, x, enc_source):
#         # 生成查询、键、和值
#         Q = self.query_linear(x)  # 音频特征生成查询
#         K = self.key_linear(enc_source)  # 面部和视角特征生成键
#         V = self.value_linear(enc_source)  # 面部和视角特征生成值
#
#         # 计算注意力分数
#         attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
#         attention_weights = F.softmax(attention_scores, dim=-1)
#
#         # 加权和值
#         out = torch.matmul(attention_weights, V)
#
#         x_ = self.offset_linear(out)
#         x = x + x_
#         return x, attention_weights
#
#
# class CrossModalAttentionModule(nn.Module):
#     """
#     交叉模态注意力模块，用于初始化模型
#     """
#
#     def __init__(self, args):
#         super(CrossModalAttentionModule, self).__init__()
#         self.cross_modal_attention = CrossModalAttention(feature_dim=args.d_model)
#
#     def forward(self, x, enc_source):
#         """
#         调用 CrossModalAttention 实现模态融合
#         """
#         x, attention_weights = self.cross_modal_attention(x, enc_source)
#         return x, attention_weights


