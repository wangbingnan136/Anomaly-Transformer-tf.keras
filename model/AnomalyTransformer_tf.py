import tensorflow as tf
from tensorflow.keras.layers import LayerNormalization, Dense, Dropout, Activation, Conv1D
from .attn_tf import AnomalyAttention, AttentionLayer
from .embed_tf import DataEmbedding, TokenEmbedding


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = Conv1D(filters=d_ff, kernel_size=1)
        self.conv2 = Conv1D(filters=d_model, kernel_size=1)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.dropout = Dropout(rate=dropout)
        self.activation = Activation("relu") if activation == "relu" else Activation("gelu")

    def call(self, x, attn_mask=None):
        new_x, attn, mask, sigma = self.attention(
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(tf.transpose(y, perm=[0, 2, 1]))))
        y = tf.transpose(self.conv2(self.dropout(y)), perm=[0, 2, 1])

        return self.norm2(x + y), attn, mask, sigma


class Encoder(tf.keras.layers.Layer):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = attn_layers
        self.norm = norm_layer

    def call(self, x, attn_mask=None):
        # x [B, L, D]
        series_list = []
        prior_list = []
        sigma_list = []
        for attn_layer in self.attn_layers:
            x, series, prior, sigma = attn_layer(x, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
            sigma_list.append(sigma)

        if self.norm is not None:
            x = self.norm(x)

        return x, series_list, prior_list, sigma_list


class AnomalyTransformer(tf.keras.Model):
    def __init__(self, win_size, enc_in, c_out, d_model=512, n_heads=8, e_layers=3, d_ff=512,
                 dropout=0.0, activation='gelu', output_attention=True):
        super(AnomalyTransformer, self).__init__()
        self.output_attention = output_attention

        # Encoding
        self.embedding = DataEmbedding(enc_in, d_model, dropout)

        # Encoder
        attn_layers = [
            EncoderLayer(
                AttentionLayer(
                    AnomalyAttention(win_size, False, attention_dropout=dropout, output_attention=output_attention),
                    d_model, n_heads),
                d_model,
                d_ff,
                dropout=dropout,
                activation=activation
            ) for l in range(e_layers)
        ]
        norm_layer = LayerNormalization(epsilon=1e-6)
        self.encoder = Encoder(attn_layers, norm_layer=norm_layer)

        self.projection = Dense(units=c_out, activation=None, use_bias=True)

    def call(self, x):
        enc_out = self.embedding(x)
        enc_out, series, prior, sigmas = self.encoder(enc_out)
        enc_out = self.projection(enc_out)

        if self.output_attention:
            return enc_out, series, prior, sigmas
        else:
        return enc_out # [B, L, D]
