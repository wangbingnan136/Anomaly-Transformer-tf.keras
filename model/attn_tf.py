import tensorflow as tf
import numpy as np
from math import sqrt

class TriangularCausalMask():
    def __init__(self, B, L):
        mask_shape = [B, 1, L, L]
        self._mask = tf.linalg.band_part(tf.ones(mask_shape, dtype=tf.bool), 0, -1)

    @property
    def mask(self):
        return self._mask


class AnomalyAttention(tf.keras.layers.Layer):
    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False):
        super(AnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = tf.keras.layers.Dropout(rate=attention_dropout)
        window_size = win_size
        self.distances = tf.zeros((window_size, window_size), dtype=tf.float32)
        for i in range(window_size):
            for j in range(window_size):
                self.distances[i][j].assign(tf.abs(i - j))

    def call(self, queries, keys, values, sigma, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = tf.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=self.device)
            scores = tf.where(attn_mask.mask, -np.inf, scores)
        attn = scale * scores

        sigma = tf.transpose(sigma, perm=[0, 2, 1])  # B L H ->  B H L
        window_size = attn.shape[-1]
        sigma = tf.sigmoid(sigma * 5) + 1e-5
        sigma = tf.pow(3, sigma) - 1
        sigma = tf.tile(sigma[:, tf.newaxis, :, :], [1, 1, 1, window_size])  # B H L L
        prior = self.distances[tf.newaxis, tf.newaxis, ...]
        prior = 1.0 / (tf.sqrt(2 * tf.constant(np.pi)) * sigma) * tf.exp(-prior ** 2 / 2 / (sigma ** 2))

        series = self.dropout(tf.nn.softmax(attn, axis=-1))
        V = tf.einsum("bhls,bshd->blhd", series, values)

        if self.output_attention:
            return (V, series, prior, sigma)
        else:
            return (V, None)


class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.inner_attention = attention
        self.query_projection = tf.keras.layers.Dense(d_keys * n_heads)
        self.key_projection = tf.keras.layers.Dense(d_keys * n_heads)
        self.value_projection = tf.keras.layers.Dense(d_values * n_heads)
        self.sigma_projection = tf.keras.layers.Dense(n_heads)
        self.out_projection = tf.keras.layers.Dense(d_model)

        self.n_heads = n_heads

    def call(self,queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        x = queries
        queries = tf.reshape(self.query_projection(queries), [B, L, H, -1])
        keys = tf.reshape(self.key_projection(keys), [B, S, H, -1])
        values = tf.reshape(self.value_projection(values), [B, S, H, -1])
        sigma = tf.reshape(self.sigma_projection(x), [B, L, H])
        out, series, prior, sigma = self.inner_attention(
            queries,
            keys,
            values,
            sigma,
            attn_mask
        )
        out = tf.reshape(out, [B, L, -1])

        return self.out_projection(self.norm(out)), series, prior, sigma
