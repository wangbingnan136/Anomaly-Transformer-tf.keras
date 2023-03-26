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
                                                use_bias=False) # default channel last
        self.tokenConv.build(input_shape=(None, None, c_in))

    def call(self, x):
        x = self.tokenConv(x)
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