import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, Dense, LSTM


class LuongDecoder(tf.keras.Model):
    """
        Luong Attention layer in Seq2Seq: https://arxiv.org/pdf/1508.04025.pdf

    """

    def __init__(self, vocab_size, embedding_size, hidden_units, **kwargs):
        super(LuongDecoder, self).__init__(**kwargs)
        self.embedding = Embedding(vocab_size, embedding_size)
        self.decode_layer_1 = LSTM(hidden_units,
                                   return_sequences=True,
                                   return_state=True,
                                   kernel_initializer="glorot_uniform")
        self.attention = LuongAttention(hidden_units=hidden_units)
        self.dense = Dense(vocab_size)

    def __call__(self, x, encoder_outs, state, *args, **kwargs):
        """
        :Input:
            - x: [batch_size, max_length]
            - encode_output: [batch_size, max_length, hidden_units]
            - State:
                + state_h: [batch_size, hidden_units] - Hidden state in encode layer
                + state_c: [batch_size, hidden_units] - Cell state in encode layer

        :return:
            - output: [batch_size, vocab_size]
            - state_h: [batch_size, hidden_units] - Current Hidden state
            - state_c: [batch_size, hidden_units] - Current Cell state
        """
        x = tf.expand_dims(x, axis=1)
        x = self.embedding(x)
        decode_outs, state_h, state_c = self.decode_layer_1(x, state)
        context_vector, att_weights = self.attention(encoder_outs, decode_outs)
        concat = tf.concat([decode_outs, context_vector], axis=-1)
        concat = tf.reshape(concat, (-1, concat.shape[2]))
        outs = self.dense(concat)
        return outs, [state_h, state_c]


class LuongAttention(Layer):

    def __init__(self, hidden_units, **kwargs):
        super(LuongAttention, self).__init__(**kwargs)
        self.Wa = Dense(hidden_units)

    def __call__(self, encoder_outs, decoder_outs, *args, **kwargs):
        score = tf.matmul(decoder_outs, self.Wa(encoder_outs), transpose_b=True)
        alignment = tf.nn.softmax(score, axis=2)
        context_vector = tf.matmul(alignment, encoder_outs)
        return context_vector, score
