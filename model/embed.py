import tensorflow as tf

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = tf.zeros([max_len, d_model], dtype=tf.float32)

        position = tf.range(0, max_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = tf.sin(position * div_term)
        pe[:, 1::2] = tf.cos(position * div_term)

        pe = pe[tf.newaxis, :]
        self.pe = tf.Variable(pe, trainable=False)

    def call(self, x):
        return self.pe[:, :tf.shape(x)[1]]


class TokenEmbedding(tf.keras.layers.Layer):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 'causal' if tf.__version__ >= '2.4.0' else 'valid'
        self.tokenConv = tf.keras.layers.Conv1D(filters=d_model,
                                                kernel_size=3,
                                                padding=padding,
                                                use_bias=False)
        self.tokenConv.build(input_shape=(None, None, c_in))
        self.tokenConv.set_weights([nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')])

    def call(self, x):
        x = tf.transpose(x, perm=[0, 2, 1])
        x = self.tokenConv(x)
        x = tf.transpose(x, perm=[0, 2, 1])
        return x


class DataEmbedding(tf.keras.layers.Layer):
    def __init__(self, c_in, d_model, dropout=0.0):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = tf.keras.layers.Dropout(rate=dropout)

    def call(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.0):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)
