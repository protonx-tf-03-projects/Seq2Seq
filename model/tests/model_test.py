import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, LSTM


class Encoder(Layer):

    def __init__(self, input_dím, embedding_dim, hidden_dim, n_layers=None):
        """
            Encoder block in Sequence to Sequence

        :param input_dím: Số lượng từ của bộ từ vựng
        :param embedding_dim: Chiều của vector embedding
        :param hidden_dim: Chiều của lớp ẩn
        :param n_layers: Số lượng layer
        """

        super(Encoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.embedding = Embedding(input_dim=input_dím,
                                   hidden_dim=embedding_dim,
                                   embeddings_initializer="he_normal",
                                   embeddings_regularizer=None,
                                   embeddings_constraint=None,
                                   mask_zero=False)
        self.encoder = LSTM(hidden_dim,
                            return_sequences=True,
                            return_state=True,
                            recurrent_initializer="he_normal")

    def call(self, x, hidden_state):
        x = self.embedding(x)
        output, last_state = self.encoder(x, hidden_state)
        if self.n_layers:
            for _ in self.n_layers - 1:
                output, last_state = self.encoder(output, hidden_state)
        return output, last_state

    def _init_hidden_state(self, batch_size):
        return tf.zeros((batch_size, self.hidden_dim))


# if __name__ == '__main__':
    # embedding_size = 1000
    # vocab_size = 10000
    # hidden_unit = 256
    # BATCH_SIZE = 32
    #
    #
    # encoder = Encoder(embedding_size, vocab_size, hidden_unit)
    # hidden_state = encoder.init_hidden_state(BATCH_SIZE)
    # tmp_outputs, last_state = encoder(tmp_x, hidden_state)
    # print(tmp_outputs.shape)
    # print(last_state.shape)