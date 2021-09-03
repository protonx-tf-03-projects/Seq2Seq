import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM


class Encode(tf.keras.Model):

    def __init__(self, vocab_size, embedding_size, hidden_units, **kwargs):
        """
            Encoder block in Sequence to Sequence

        :param vocab_size: Số lượng từ của bộ từ vựng đầu vào
        :param embedding_size: Chiều của vector embedding
        :param hidden_units: Chiều của lớp ẩn
        """
        super(Encode, self).__init__(**kwargs)

        self.hidden_units = hidden_units

        self.embedding = Embedding(vocab_size, embedding_size)
        self.encode_layer_1 = LSTM(hidden_units,
                                   return_sequences=True,
                                   return_state=True,
                                   kernel_initializer="glorot_uniform")

    def __call__(self, x, *args, **kwargs):
        """
        :Input:
            - x: [batch_size, max_length]

        :return:
            - output: [batch_size, embedding_dim, Hidden_unites]
            - state_h: [batch_size, hidden_units] - Current Hidden state
            - state_c: [batch_size, hidden_units] - Current Cell state
        """
        first_state = self.init_hidden_state(x.shape[0])
        encode = self.embedding(x)
        encode, state_h, state_c = self.encode_layer_1(encode, first_state, **kwargs)
        return encode, [state_h, state_c]

    def init_hidden_state(self, batch_size):
        return [tf.zeros([batch_size, self.hidden_units]), tf.zeros([batch_size, self.hidden_units])]
