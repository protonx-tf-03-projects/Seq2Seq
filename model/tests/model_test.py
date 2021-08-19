import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, LSTM, Input
from tensorflow.keras.models import Model


class Seq2SeqEncode(Layer):

    def __init__(self, vocab_size, embedding_size, hidden_units, n_layers=None):
        """
            Encoder block in Sequence to Sequence

        :param vocab_size: Số lượng từ của bộ từ vựng đầu vào
        :param embedding_size: Chiều của vector embedding
        :param hidden_units: Chiều của lớp ẩn
        """
        super(Seq2SeqEncode, self).__init__()

        self.hidden_units = hidden_units

        self.embedding = Embedding(vocab_size, embedding_size)
        self.encode_layer_1 = LSTM(hidden_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer="he_normal")
        self.encode_layer_2 = LSTM(hidden_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer="he_normal")

    def __call__(self, x, state):
        """
        :input:
            - x: [batch_size, max_length]

        :return:
            - output: [batch_size, embedding_dim, Hidden_unites]
            - state_h: [batch_size, hidden_units] - Current Hidden state
            - state_c: [batch_size, hidden_units] - Current Cell state
        """
        encode = self.embedding(x)
        # encode = self.encode_layer_1(encode, state)
        encode, state_h, state_c = self.encode_layer_2(encode, state)
        return encode, [state_h, state_c]

    def _init_hidden_state_(self, batch_size):
        return [tf.zeros([batch_size, self.hidden_units]), tf.zeros([batch_size, self.hidden_units])]


class Seq2SeqDecode(Layer):
    def __init__(self, vocab_size, embedding_size, hidden_units, n_layers=None):
        """
            Decoder block in Sequence to Sequence

        :param vocab_size: Số lượng từ của bộ từ vựng dự đoán
        :param embedding_size: Chiều của vector embedding
        :param hidden_units: Chiều của lớp ẩn
        """
        super(Seq2SeqDecode, self).__init__()

        self.embedding = Embedding(vocab_size, embedding_size)
        self.decode_layer_1 = LSTM(hidden_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer="he_normal")
        self.decode_layer_2 = LSTM(hidden_units,
                                   return_state=True,
                                   recurrent_initializer="he_normal")
        self.dense = tf.keras.layers.Dense(vocab_size, activation="softmax")

    def __call__(self, x, state, **kwargs):
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
        # decode = self.decode_layer_1(decode, state)
        decode, state_h, state_c = self.decode_layer_2(decode, state)
        output_decode = self.dense(decode)
        return output_decode, [state_h, state_c]
