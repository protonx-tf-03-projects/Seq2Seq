import os
import numpy as np
import tensorflow as tf

from tqdm import tqdm
from data import DatasetLoader
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model.tests.model_test import Seq2SeqEncode, Seq2SeqDecode, EncoderDecoder


class MaskedSoftmaxCELoss(tf.keras.losses.Loss):
    """
        The softmax cross-entropy loss with masks.

        Focus calculate positive loss between y_pred and y_true points
        (Tính giá trị mất mát tập chung các vị trí xuất hiện (Y_hat) từ giống với giá trị gốc (Y_true):)

        Ex: Simple Mask_matrix
            input:
                pred_matrix = [1, 25, 1445, 105, 5, 4, 8, 2]
                true_matrix = [0, 20, 1456, 145, 2, 0, 0, 0]

            Calculate loss with mask matrix for flowing true_matrix:
            mask_matrix = [0, 1, 1, 1, 1, 0, 0, 0]

            loss = true_matrix - pred_matrix = [1, 0, 11, 40, 0, 4, 8, 2]
            ==> loss = loss * mask_matrix = [0, 5, 11, 40, 3, 0, 0, 0]
    """

    def __init__(self, valid_len):
        super().__init__(reduction='none')
        self.valid_len = valid_len  # valid_len shape: (batch_size,)

    def call(self, label, pred):
        """
        :param label: shape (batch_size, max_length, vocab_size)
        :param pred: shape (batch_size, max_length)
        
        :return: weighted_loss: shape (batch_size, max_length)
        """

        weights_mask = 1 - np.equal(label, 0)

        label_one_hot = tf.one_hot(label, depth=pred.shape[-1])

        unweighted_loss = tf.keras.losses.CategoricalCrossentropy(
            from_logits=True, reduction='none')(label_one_hot, pred)

        weighted_loss = tf.reduce_mean(unweighted_loss * weights_mask)
        return weighted_loss


class SequenceToSequence:
    def __init__(self,
                 inp_lang,
                 tar_lang,
                 embedding_size=64,
                 hidden_units=256,
                 test_split_size=0.05,
                 max_length=64,
                 epochs=400,
                 batch_size=64):
        self.inp_lang = inp_lang
        self.tar_lang = tar_lang
        self.embedding_size = embedding_size
        self.hidden_units = hidden_units

        self.max_length = max_length
        self.test_split_size = test_split_size

        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs

    def train(self):
        inp_tensor, tar_tensor, caches = DatasetLoader(self.inp_lang, self.tar_lang).build_dataset()

        net = EncoderDecoder(inp_vocab_size=caches[0].vocab_size,
                             tar_vocab_size=caches[1].vocab_size,
                             embedding_size=self.embedding_size,
                             hidden_units=self.hidden_units,
                             batch_size=self.BATCH_SIZE)

        optimizer = tf.keras.optimizers.Adam()

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
            loss = 0
            for batch_size, (x, y) in tqdm(enumerate(train_ds.take(N_BATCH)), total=N_BATCH):
                with tf.GradientTape() as tape:
                    decode_out = net(x, y)
                    loss += MaskedSoftmaxCELoss(y)(y, decode_out)

                train_vars = net.trainable_variables
                grads = tape.gradient(loss, train_vars)
                optimizer.apply_gradients(zip(grads, train_vars))

            print(f'\nLoss: {loss}')


if __name__ == "__main__":
    parser = ArgumentParser()

    # FIXME
    # Arguments users used when running command lines
    parser.add_argument("--inp-lang", required=True, type=str)
    parser.add_argument("--tar-lang", required=True, type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--embedding_size", default=64, type=int)
    parser.add_argument("--hidden_units", default=256, type=int)
    parser.add_argument("--test_split_size", default=0.1, type=int)

    home_dir = os.getcwd()
    args = parser.parse_args()

    # FIXME
    # Project Description

    print('---------------------Welcome to ${Hợp tác xã kiên trì}-------------------')
    print('Github: ${Xunino}')
    print('Email: ${ndlinh.ai@gmail.com}')
    print('---------------------------------------------------------------------')
    print(f'Training ${SequenceToSequence} model with hyper-params:')  # FIXME
    print(args)
    print('===========================')

    # FIXME
    # Do Training
    # SequenceToSequence("dataset/train.en.txt", "dataset/train.vi.txt").train()
    SequenceToSequence(inp_lang=args.inp_lang,
                       tar_lang=args.tar_lang,
                       batch_size=args.batch_size,
                       embedding_size=args.embedding_size,
                       hidden_units=args.hidden_units,
                       test_split_size=args.test_split_size,
                       epochs=args.epochs).train()
