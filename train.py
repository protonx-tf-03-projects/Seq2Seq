import os
import time

import numpy as np
import tensorflow as tf

from argparse import ArgumentParser

from tensorflow.keras.preprocessing.sequence import pad_sequences
from model.tests.model_test import Seq2SeqEncode, Seq2SeqDecode
from data import DatasetLoader
from sklearn.model_selection import train_test_split


class MaskedSoftmaxCELoss(tf.keras.losses.Loss):
    """The softmax cross-entropy loss with masks."""

    def __init__(self, valid_len):
        super().__init__(reduction='none')
        self.valid_len = valid_len  # `valid_len` shape: (`batch_size`,)

    def call(self, label, pred):
        # `pred` shape: (`batch_size`, `num_steps`, `vocab_size`)
        # `label` shape: (`batch_size`, `num_steps`)

        weights_mask = tf.ones_like(label, dtype=tf.float32)
        weights_mask = self.sequence_mask(weights_mask, self.valid_len)

        label_one_hot = tf.one_hot(label, depth=pred.shape[-1])

        unweighted_loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, reduction='none')(label_one_hot, pred)

        weighted_loss = tf.reduce_mean((unweighted_loss * weights_mask), axis=1)
        return weighted_loss

    def sequence_mask(self, X, valid_len, value=0):
        """Mask irrelevant entries in sequences."""
        maxlen = X.shape[1]
        mask = tf.range(start=0, limit=maxlen,
                        dtype=tf.float32)[None, :] < tf.cast(
            valid_len[:, None], dtype=tf.float32)

        if len(X.shape) == 3:
            return tf.where(tf.expand_dims(mask, axis=-1), X, value)
        else:
            return tf.where(mask, X, value)


class SequenceToSequence:
    def __init__(self,
                 vocab_1,
                 vocab_2,
                 embedding_size=64,
                 hidden_units=256,
                 test_split_size=0.05,
                 max_length=64,
                 epochs=400,
                 batch_size=64):
        self.vocab_1 = vocab_1
        self.vocab_2 = vocab_2
        self.embedding_size = embedding_size
        self.hidden_units = hidden_units

        self.max_length = max_length
        self.test_split_size = test_split_size

        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs

    def loss_function(self, y_true, y_pred):
        mask = 1 - np.equal(y_true, 0)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_true, y_pred)
        return tf.reduce_mean(loss * mask)

    def train(self):
        inp_tensor, tar_tensor, caches = DatasetLoader(self.vocab_1, self.vocab_2).build_dataset()

        self.inp_lang, self.tar_lang = caches

        self.encode = Seq2SeqEncode(vocab_size=self.inp_lang.vocab_size,
                                    embedding_size=self.embedding_size,
                                    hidden_units=self.hidden_units)

        self.decode = Seq2SeqDecode(vocab_size=self.tar_lang.vocab_size,
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

        self.N_BATCH = train_x.shape[0] // self.BATCH_SIZE

        for epoch in range(self.EPOCHS):
            for batch_size, (x, y) in enumerate(train_ds.take(self.N_BATCH)):
                with tf.GradientTape() as tape:
                    hidden_state = self.encode._init_hidden_state_(self.BATCH_SIZE)
                    _, last_state = self.encode(x, hidden_state)

                    """
                        If first sentences not have <sos>:
                            sos = tf.reshape(tf.constant([self.tar_lang.word2id['<sos>']] * y.shape[0]), shape=(-1, 1))
                            dec_input = np.concatenate([sos, y[:, :-1]], 1)
                    """
                    dec_input = y
                    decode_out, _ = self.decode(dec_input, last_state)

                    loss = MaskedSoftmaxCELoss(y)(y, decode_out)

                train_vars = self.encode.trainable_variables + self.decode.trainable_variables
                grads = tape.gradient(loss, train_vars)
                optimizer.apply_gradients(zip(grads, train_vars))

            print(f'loss {tf.reduce_sum(loss)}')


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
