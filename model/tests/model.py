import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, Dense, LSTM


class Seq2SeqEncode(tf.keras.Model):

    def __init__(self, vocab_size, embedding_size, hidden_units, **kwargs):
        """
            Encoder block in Sequence to Sequence

        :param vocab_size: Số lượng từ của bộ từ vựng đầu vào
        :param embedding_size: Chiều của vector embedding
        :param hidden_units: Chiều của lớp ẩn
        """
        super(Seq2SeqEncode, self).__init__(**kwargs)

        self.hidden_units = hidden_units

        self.embedding = Embedding(vocab_size, embedding_size)
        self.encode_layer_1 = LSTM(hidden_units,
                                   recurrent_dropout=0.2,
                                   return_sequences=True,
                                   return_state=True,
                                   kernel_initializer="glorot_uniform")
        self.dense = Dense(hidden_units)

    def __call__(self, x, first_state, *args, **kwargs):
        """
        :input:
            - x: [batch_size, max_length]

        :return:
            - output: [batch_size, embedding_dim, Hidden_unites]
            - state_h: [batch_size, hidden_units] - Current Hidden state
            - state_c: [batch_size, hidden_units] - Current Cell state
        """
        encode = self.embedding(x)
        encode, state_h, state_c = self.encode_layer_1(encode, first_state, **kwargs)
        encode = self.dense(encode, **kwargs)
        return encode, [state_h, state_c]

    def init_hidden_state(self, batch_size):
        return [tf.zeros([batch_size, self.hidden_units]), tf.zeros([batch_size, self.hidden_units])]


class Seq2SeqDecode(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, hidden_units, **kwargs):
        """
            Decoder block in Sequence to Sequence

        :param vocab_size: Số lượng từ của bộ từ vựng dự đoán
        :param embedding_size: Chiều của vector embedding
        :param hidden_units: Chiều của lớp ẩn
        """
        super(Seq2SeqDecode, self).__init__(**kwargs)

        self.embedding = Embedding(vocab_size, embedding_size)
        self.decode_layer_1 = LSTM(hidden_units,
                                   recurrent_dropout=0.2,
                                   return_sequences=True,
                                   return_state=True,
                                   kernel_initializer="glorot_uniform")
        self.dense = Dense(vocab_size)

    def __call__(self, x, state, *args, **kwargs):
        """
        :input:
            - x: [batch_size, max_length]
            - State:
                + state_h: [batch_size, hidden_units] - Hidden state in encode layer
                + state_c: [batch_size, hidden_units] - Cell state in encode layer

        :return:
            - output: [batch_size, embedding_dim, Hidden_unites]
            - state_h: [batch_size, hidden_units] - Current Hidden state
            - state_c: [batch_size, hidden_units] - Current Cell state
        """

        decode = self.embedding(x)  # [Batch_size, vocab_length, Embedding_size]
        decode, state_h, state_c = self.decode_layer_1(decode, state, **kwargs)
        output_decode = self.dense(decode)
        return output_decode, [state_h, state_c]


class Bahdanau_Attention(Layer):
    """
        Bahdanua attention paper: https://arxiv.org/pdf/1409.0473.pdf
    """

    def __init__(self, hidden_units, **kwargs):
        super(Bahdanau_Attention, self).__init__(**kwargs)
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


class BahdanauSeq2SeqDecode(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, hidden_units, **kwargs):
        """
            Decoder vs Attention block in Sequence to Sequence

        :param vocab_size: Số lượng từ của bộ từ vựng dự đoán
        :param embedding_size: Chiều của vector embedding
        :param hidden_units: Chiều của lớp ẩn
        """
        super(BahdanauSeq2SeqDecode, self).__init__(**kwargs)

        self.embedding = Embedding(vocab_size, embedding_size)
        self.decode_layer_1 = LSTM(hidden_units,
                                   recurrent_dropout=0.2,
                                   return_sequences=True,
                                   return_state=True,
                                   kernel_initializer="glorot_uniform")
        self.attention = Bahdanau_Attention(hidden_units=hidden_units)
        self.dense = Dense(vocab_size)

    def __call__(self, x, encoder_outs, state, *args, **kwargs):
        """
        :input:
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


class LuongSeq2SeqDecoder(tf.keras.Model):
    """
        Luong Attention layer in Seq2Seq: https://arxiv.org/pdf/1508.04025.pdf

    """

    def __init__(self, vocab_size, embedding_size, hidden_units, **kwargs):
        super(LuongSeq2SeqDecoder, self).__init__(**kwargs)
        self.embedding = Embedding(vocab_size, embedding_size)
        self.decode_layer_1 = LSTM(hidden_units,
                                   recurrent_dropout=0.2,
                                   return_sequences=True,
                                   return_state=True,
                                   kernel_initializer="glorot_uniform")
        self.attention = LuongAttention(hidden_units=hidden_units)
        self.dense = Dense(vocab_size)

    def __call__(self, x, encoder_outs, state, *args, **kwargs):
        """
        :input:
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
