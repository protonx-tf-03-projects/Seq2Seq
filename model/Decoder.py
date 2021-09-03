import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, LSTM


class Decode(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, hidden_units, **kwargs):
        """
            Decoder block in Sequence to Sequence

        :param vocab_size: Số lượng từ của bộ từ vựng dự đoán
        :param embedding_size: Chiều của vector embedding
        :param hidden_units: Chiều của lớp ẩn
        """
        super(Decode, self).__init__(**kwargs)

        self.embedding = Embedding(vocab_size, embedding_size)
        self.decode_layer_1 = LSTM(hidden_units,
                                   return_sequences=True,
                                   return_state=True,
                                   kernel_initializer="glorot_uniform")
        self.dense = Dense(vocab_size)

    def __call__(self, x, state, *args, **kwargs):
        """
        :Input:
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
