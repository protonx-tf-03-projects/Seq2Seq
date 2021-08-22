import os
import time

import numpy as np
import tensorflow as tf

from tqdm import tqdm
from data import DatasetLoader
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model.tests.model_test import EncoderDecoder


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

    def __init__(self):
        super().__init__()

    def call(self, label, pred):
        """
        :param label: shape (batch_size, max_length, vocab_size)
        :param pred: shape (batch_size, max_length)

        :return: weighted_loss: shape (batch_size, max_length)
        """

        weights_mask = 1 - np.equal(label, 0)
        unweighted_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(label, pred)
        weighted_loss = tf.reduce_mean(unweighted_loss * weights_mask)
        return weighted_loss


class SequenceToSequence:
    def __init__(self,
                 inp_lang,
                 tar_lang,
                 embedding_size=64,
                 hidden_units=256,
                 test_split_size=0.05,
                 max_length=32,
                 epochs=400,
                 batch_size=128):
        self.inp_lang = inp_lang
        self.tar_lang = tar_lang
        self.embedding_size = embedding_size
        self.hidden_units = hidden_units

        self.max_length = max_length
        self.test_split_size = test_split_size

        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs

        self.optimizer = tf.keras.optimizers.Adam()

    def training(self, net, train_ds, test_ds, N_BATCH):
        for epoch in range(self.EPOCHS):
            loss = 0
            for batch_size, (x, y) in tqdm(enumerate(train_ds.take(N_BATCH)), total=N_BATCH):
                with tf.GradientTape() as tape:
                    bos = tf.reshape(tf.constant([self.caches[1].word2id['<sos>']] * y.shape[0]), shape=(-1, 1))
                    dec_input = tf.concat([bos, y[:, :-1]], 1)  # Teacher forcing
                    decode_out = net(x, dec_input, training=True)
                    loss += MaskedSoftmaxCELoss()(y, decode_out)

                train_vars = net.trainable_variables
                grads = tape.gradient(loss, train_vars)
                self.optimizer.apply_gradients(zip(grads, train_vars))
            print("\n=================================================================")
            print(f'Epoch {epoch + 1} -- Loss: {loss}')

            # Evaluate
            self.evaluation(net, test_ds)

    def evaluation(self, model, test_ds):
        """
        :param model: Seq2Seq
        :param test_ds: (inp_vocab, tar_vocab)
        :return:
        """
        # Preprocessing testing data
        for test_, test_y in test_ds.take(1):
            test_x = np.expand_dims(test_.numpy(), axis=0)
            _, last_state = model.encoder(test_x, training=False)

            dec_X = np.expand_dims(np.array([self.caches[1].word2id['<sos>']]), axis=0)
            sentence = []
            for _ in range(30):
                output, last_state = model.decoder(dec_X, last_state, training=False)
                output = np.argmax(output, axis=2)
                sentence.append(output[0][0])

                if output[0][0] == self.caches[1].word2id["<eos>"]:
                    break

            print("\n-----------------------------------------------------------------")
            print("Input: ", self.caches[0].vector_to_sentence(test_.numpy()))
            print("Predicted: ", self.caches[1].vector_to_sentence(sentence))
            print("Target: ", self.caches[1].vector_to_sentence(test_y.numpy()))
            print("=================================================================\n")

    def run(self):
        inp_tensor, tar_tensor, self.caches = DatasetLoader(self.inp_lang, self.tar_lang).build_dataset()

        net = EncoderDecoder(inp_vocab_size=self.caches[0].vocab_size,
                             tar_vocab_size=self.caches[1].vocab_size,
                             embedding_size=self.embedding_size,
                             hidden_units=self.hidden_units)

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
        test_ds = tf.data.Dataset.from_tensor_slices((test_x, test_y))

        N_BATCH = train_x.shape[0] // self.BATCH_SIZE

        # Training
        self.training(net, train_ds, test_ds, N_BATCH)


if __name__ == "__main__":
    parser = ArgumentParser()

    # FIXME
    # Arguments users used when running command lines
    # parser.add_argument("--inp-lang", required=True, type=str)
    # parser.add_argument("--tar-lang", required=True, type=str)
    # parser.add_argument("--batch_size", default=64, type=int)
    # parser.add_argument("--epochs", default=1000, type=int)
    # parser.add_argument("--embedding_size", default=64, type=int)
    # parser.add_argument("--hidden_units", default=256, type=int)
    # parser.add_argument("--test_split_size", default=0.1, type=int)

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
    SequenceToSequence("dataset/train.en.txt", "dataset/train.vi.txt").run()
    # SequenceToSequence(inp_lang=args.inp_lang,
    #                    tar_lang=args.tar_lang,
    #                    batch_size=args.batch_size,
    #                    embedding_size=args.embedding_size,
    #                    hidden_units=args.hidden_units,
    #                    test_split_size=args.test_split_size,
    #                    epochs=args.epochs).run()
