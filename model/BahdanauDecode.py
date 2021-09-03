import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, Dense, LSTM


class BahdanauDecode(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, hidden_units, **kwargs):
        """
            Decoder vs Attention block in Sequence to Sequence

        :param vocab_size: Số lượng từ của bộ từ vựng dự đoán
        :param embedding_size: Chiều của vector embedding
        :param hidden_units: Chiều của lớp ẩn
        """
        super(BahdanauDecode, self).__init__(**kwargs)

        self.embedding = Embedding(vocab_size, embedding_size)
        self.decode_layer_1 = LSTM(hidden_units,
                                   return_sequences=True,
                                   return_state=True,
                                   kernel_initializer="glorot_uniform")
        self.attention = BahdanauAttention(hidden_units=hidden_units)
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
        x = self.embedding(x)  # [Batch_size, vocab_length, Embedding_size]
        context_vector, attention_weight = self.attention(encoder_outs, state)
        context_vector = tf.expand_dims(context_vector, axis=1)
        decode_inp = tf.concat([x, context_vector], axis=-1)  # vocab_length
        decode, state_h, state_c = self.decode_layer_1(decode_inp, state, **kwargs)
        decode = tf.reshape(decode, (-1, decode.shape[2]))
        decode_output = self.dense(decode)
        return decode_output, [state_h, state_c]


class BahdanauAttention(Layer):
    """
        Bahdanua attention paper: https://arxiv.org/pdf/1409.0473.pdf
    """

    def __init__(self, hidden_units, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)
        self.weight_output_encoder = Dense(hidden_units)
        self.weight_state_h = Dense(hidden_units)
        self.score = Dense(1)

    def __call__(self, encode_output, state, *args, **kwargs):
        """
        :param encode_output: [batch_size, max_len, hidden_units]
        :param state: (state_h)
        :param args:
        :param kwargs:
        :return:
            context_vector: (batch_size, 1, hidden_unites)
        """
        state_h = tf.expand_dims(state[0], axis=1)
        state_h = self.weight_state_h(state_h)
        encode_out = self.weight_output_encoder(encode_output)
        score = self.score(
            tf.nn.tanh(state_h + encode_out)
        )
        score = tf.nn.softmax(score, axis=1)  # focus importance thing on sentence (in vocab_length dimension)
        context_vector = score * encode_out
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, score
