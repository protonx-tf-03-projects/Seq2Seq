import os
import time

import numpy as np
import tensorflow as tf

from argparse import ArgumentParser

from keras import Input, Model
from keras.layers import Dense, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model.tests.model_test import Seq2SeqEncode, Seq2SeqDecode
from data import DatasetLoader
from sklearn.model_selection import train_test_split


class SequenceToSequence:
    def __init__(self,
                 vocab_1,
                 vocab_2,
                 embedding_size=64,
                 hidden_units=256,
                 test_split_size=0.05,
                 max_length=64,
                 epochs=10,
                 batch_size=8):
        self.vocab_1 = vocab_1
        self.vocab_2 = vocab_2
        self.embedding_size = embedding_size
        self.hidden_units = hidden_units

        self.max_length = max_length
        self.test_split_size = test_split_size

        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs

    def BuildTokenizer(self, text, num_words):
        token = Tokenizer(num_words)
        token.fit_on_texts(text)
        return token, token.texts_to_sequences(text)

    def loss_function(self, y_true, y_pred):
        mask = 1 - np.equal(y_true, 0)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_true, y_pred)
        return tf.reduce_mean(loss * mask)

    def train(self):
        inp_tensor, tar_tensor, caches = DatasetLoader(self.vocab_1, self.vocab_2).build_dataset()

        inp_lang, tar_lang = caches

        encode = Seq2SeqEncode(vocab_size=inp_lang.vocab_size,
                               embedding_size=self.embedding_size,
                               hidden_units=self.hidden_units)
        hidden_state = encode._init_hidden_state_(self.BATCH_SIZE)

        decode = Seq2SeqDecode(vocab_size=tar_lang.vocab_size,
                               embedding_size=self.embedding_size,
                               hidden_units=self.hidden_units)

        optimizer = tf.keras.optimizers.Adam(0.001)

        padded_sequences_vi = pad_sequences(inp_tensor,
                                            maxlen=self.max_length,
                                            padding="post",
                                            truncating="post")
        padded_sequences_en = pad_sequences(tar_tensor,
                                            maxlen=self.max_length,
                                            padding="post",
                                            truncating="post")

        train_x, test_x, train_y, test_y = train_test_split(padded_sequences_vi,
                                                            padded_sequences_en,
                                                            test_size=self.test_split_size)

        train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(self.BATCH_SIZE)
        test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(self.BATCH_SIZE)

        N_BATCH = train_x.shape[0] // self.BATCH_SIZE

        for epoch in range(self.EPOCHS):
            total_loss = 0
            for batch_size, (x, y) in enumerate(train_ds.take(N_BATCH)):
                loss = 0
                with tf.GradientTape() as tape:
                    _, last_state = encode(x, hidden_state)
                    sos = np.array([tar_lang.word2id["<sos>"]] * self.BATCH_SIZE).reshape(-1, 1)
                    dec_input = np.concatenate([sos, y[:, :-1]], 1)

                    for i in range(1, y.shape[1]):
                        decode_out, _ = decode(dec_input, last_state)
                        loss += self.loss_function(y[:, i], decode_out)

                    train_vars = encode.trainable_variables + decode.trainable_variables
                    grads = tape.gradient(loss, train_vars)
                    optimizer.apply_gradients(zip(grads, train_vars))

                total_loss += loss

            print(total_loss.numpy())


if __name__ == '__main__':
    SequenceToSequence("dataset/train.en.txt", "dataset/train.vi.txt").train()
# if __name__ == "__main__":
#     parser = ArgumentParser()
#
#     # FIXME
#     # Arguments users used when running command lines
#     parser.add_argument("--batch-size", default=64, type=int)
#     parser.add_argument("--epochs", default=1000, type=int)
#
#     home_dir = os.getcwd()
#     args = parser.parse_args()
#
#     # FIXME
#     # Project Description
#
#     print('---------------------Welcome to ${name}-------------------')
#     print('Github: ${accout}')
#     print('Email: ${email}')
#     print('---------------------------------------------------------------------')
#     print('Training ${name} model with hyper-params:')  # FIXME
#     print('===========================')
#
#     # FIXME
#     # Do Prediction
