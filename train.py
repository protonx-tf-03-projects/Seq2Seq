import os
import time
import collections

import numpy as np
import tensorflow as tf

from tqdm import tqdm
from data import DatasetLoader
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model.tests.model_test import EncoderDecoder, AttentionEncoderDecoder, Seq2SeqEncode, Seq2SeqDecode, \
    AttentionSeq2SeqDecode


class Bleu_score:
    """
        We can evaluate a predicted sequence by comparing it with the label sequence.
        BLEU (Bilingual Evaluation Understudy) "https://aclanthology.org/P02-1040.pdf",
        though originally proposed for evaluating machine translation results,
        has been extensively used in measuring the quality of output sequences for different applications.
        In principle, for any n-grams in the predicted sequence, BLEU evaluates whether this n-grams appears
        in the label sequence.
    """

    def __init__(self):
        super().__init__()

    def remove_oov(self, sentence):
        return [i for i in sentence.split(" ") if i not in ["<eos>", "<sos>"]]

    def __call__(self, predicted_sentence, target_sentences, n_grams=3):
        predicted_sentence = self.remove_oov(predicted_sentence)
        target_sentences = self.remove_oov(target_sentences)
        pred_length = len(predicted_sentence)
        target_length = len(target_sentences)

        if pred_length == 0 and pred_length != target_length:
            return 0
        else:
            score = np.exp(np.minimum(0, 1 - target_length / pred_length))
            for k in range(1, n_grams + 1):
                label_subs = collections.defaultdict(int)
                for i in range(target_length - k + 1):
                    label_subs[" ".join(target_sentences[i:i + k])] += 1

                num_matches = 0
                for i in range(pred_length - k + 1):
                    if label_subs[" ".join(predicted_sentence[i:i + k])] > 0:
                        label_subs[" ".join(predicted_sentence[i:i + k])] -= 1
                        num_matches += 1
                score *= np.power(num_matches / (pred_length - k + 1), np.power(0.5, k))
            return score


class MaskedSoftmaxCELoss(tf.keras.losses.Loss):
    """
        The softmax cross-entropy loss with masks.

        Tính giá trị mất mát tập chung các vị trí xuất hiện (Y_hat) từ giống với giá trị gốc (Y_true):

        Ví dụ cơ bản:
            Đầu vào:
                pred_matrix = [1, 25, 1445, 105, 5, 4, 8, 2]
                true_matrix = [0, 20, 1456, 145, 2, 0, 0, 0]

            Chỗ nào có giá trị khác 0 tại true_matrix gán bằng giá trị 1:
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
                 inp_lang_path,
                 tar_lang_path,
                 embedding_size=64,
                 hidden_units=256,
                 test_split_size=0.005,
                 epochs=400,
                 batch_size=128,
                 min_sentence=10,
                 max_sentence=14,
                 mode_training="attention",
                 debug=False):
        self.inp_lang_path = inp_lang_path
        self.tar_lang_path = tar_lang_path
        self.embedding_size = embedding_size
        self.hidden_units = hidden_units

        self.max_length = max_sentence
        self.test_split_size = test_split_size
        self.min_sentence = min_sentence
        self.max_sentence = max_sentence

        self.BATCH_SIZE = batch_size
        self.EPOCHS = epochs
        self.mode_training = mode_training
        self.debug = debug

        # Load dataset
        self.inp_tensor, self.tar_tensor, self.inp_lang, self.tar_lang = DatasetLoader(self.inp_lang_path,
                                                                                       self.tar_lang_path,
                                                                                       self.min_sentence,
                                                                                       self.max_sentence).build_dataset()
        # Initialize optimizer
        self.optimizer = tf.keras.optimizers.Adam()
        # Initialize encoder
        self.encoder = Seq2SeqEncode(self.inp_lang.vocab_size,
                                     self.embedding_size,
                                     self.hidden_units)
        # Initialize first state
        self.first_state = self.encoder.init_hidden_state(self.BATCH_SIZE)
        # Initialize decoder
        self.decoder = Seq2SeqDecode(self.tar_lang.vocab_size,
                                     self.embedding_size,
                                     self.hidden_units)
        # Initialize decoder with attention
        self.decoder_attention = AttentionSeq2SeqDecode(self.tar_lang.vocab_size,
                                                        self.embedding_size,
                                                        self.hidden_units)

    def training(self, train_ds, N_BATCH):
        for epoch in range(self.EPOCHS):
            loss = 0
            for batch_size, (x, y) in tqdm(enumerate(train_ds.batch(self.BATCH_SIZE).take(N_BATCH)), total=N_BATCH):
                with tf.GradientTape() as tape:
                    encoder_outs, last_state = self.encoder(x, self.first_state)
                    sos = tf.reshape(tf.constant([self.tar_lang.word2id['<sos>']] * self.BATCH_SIZE), shape=(-1, 1))
                    dec_input = tf.concat([sos, y[:, :-1]], 1)  # Teacher forcing
                    decode_out, _ = self.decoder(dec_input, last_state)
                    loss += MaskedSoftmaxCELoss()(y, decode_out)

                train_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
                grads = tape.gradient(loss, train_vars)
                self.optimizer.apply_gradients(zip(grads, train_vars))

            bleu_score = self.evaluation(train_ds, self.debug)
            print("\n=================================================================")
            print(f'Epoch {epoch + 1} -- Loss: {loss} -- Bleu_score: {bleu_score}')
            print("=================================================================\n")

    def training_with_attention(self, train_ds, N_BATCH):
        for epoch in range(self.EPOCHS):
            total_loss = 0
            for batch_size, (x, y) in tqdm(enumerate(train_ds.batch(self.BATCH_SIZE).take(N_BATCH)), total=N_BATCH):
                loss = 0
                with tf.GradientTape() as tape:
                    encoder_outs, last_state = self.encoder(x, self.first_state)

                    dec_input = tf.constant([self.tar_lang.word2id['<sos>']] * self.BATCH_SIZE)
                    for i in range(1, y.shape[1]):
                        decode_out, _ = self.decoder_attention(dec_input, encoder_outs, last_state)
                        loss += MaskedSoftmaxCELoss()(y[:, i], decode_out)
                        dec_input = y[:, i]

                train_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
                grads = tape.gradient(loss, train_vars)
                self.optimizer.apply_gradients(zip(grads, train_vars))
                total_loss += loss

            bleu_score = self.evaluation_with_attention(train_ds, self.debug)
            print("\n=================================================================")
            print(f'Epoch {epoch + 1} -- Loss: {total_loss} -- Bleu_score: {bleu_score}')
            print("=================================================================\n")

    def evaluation(self, test_ds, debug=False):
        """
        :param model: Seq2Seq
        :param test_ds: (inp_vocab, tar_vocab)
        :param (inp_lang, tar_lang)
        :return:
        """
        # Preprocessing testing data
        score = 0.0
        test_ds_len = int(len(test_ds) * self.test_split_size)
        count = 0
        for test_, test_y in test_ds.shuffle(buffer_size=1, seed=1).take(test_ds_len):
            test_x = np.expand_dims(test_.numpy(), axis=0)
            first_state = self.encoder.init_hidden_state(batch_size=1)
            _, last_state = self.encoder(test_x, first_state, training=False)

            input_decode = tf.reshape(tf.constant([self.tar_lang.word2id['<sos>']]), shape=(-1, 1))
            sentence = []
            for _ in range(self.tar_lang.max_len):
                output, last_state = self.decoder(input_decode, last_state, training=False)
                output = tf.argmax(output, axis=2).numpy()
                input_decode = output
                sentence.append(output[0][0])
            score += Bleu_score()(self.tar_lang.vector_to_sentence(sentence),
                                  self.tar_lang.vector_to_sentence(test_y.numpy()))
            if debug and count <= 5:
                print("-----------------------------------------------------------------")
                print("Input    : ", self.inp_lang.vector_to_sentence(test_.numpy()))
                print("Predicted: ", self.tar_lang.vector_to_sentence(sentence))
                print("Target   : ", self.tar_lang.vector_to_sentence(test_y.numpy()))
                print("=================================================================")
            count += 1
        return score / test_ds_len

    def evaluation_with_attention(self, test_ds, debug=True):
        """
        :param model: Seq2Seq
        :param test_ds: (inp_vocab, tar_vocab)
        :param caches: (inp_lang, tar_lang)
        :return:
        """
        # Preprocessing testing data
        score = 0.0
        test_ds_len = int(len(test_ds) * self.test_split_size)
        count = 0
        for test_, test_y in test_ds.shuffle(buffer_size=1, seed=1).take(test_ds_len):
            test_x = np.expand_dims(test_.numpy(), axis=0)
            first_state = self.encoder.init_hidden_state(batch_size=1)
            encode_outs, last_state = self.encoder(test_x, first_state, training=False)

            input_decode = np.array([self.tar_lang.word2id['<eos>']])
            sentence = []
            for _ in range(self.tar_lang.max_len):
                output, last_state = self.decoder_attention(input_decode, encode_outs, last_state, training=False)
                output = tf.argmax(output, axis=1).numpy()
                input_decode = output
                sentence.append(output[0])

            score += Bleu_score()(self.tar_lang.vector_to_sentence(sentence),
                                  self.tar_lang.vector_to_sentence(test_y.numpy()))
            if debug and count <= 5:
                print("-----------------------------------------------------------------")
                print("Input    : ", self.inp_lang.vector_to_sentence(test_.numpy()))
                print("Predicted: ", self.tar_lang.vector_to_sentence(sentence))
                print("Target   : ", self.tar_lang.vector_to_sentence(test_y.numpy()))
                print("=================================================================")
            count += 1
        return score / test_ds_len

    def run(self):
        # Padding in sequences
        train_x = pad_sequences(self.inp_tensor,
                                maxlen=self.max_length,
                                padding="post",
                                truncating="post")
        train_y = pad_sequences(self.tar_tensor,
                                maxlen=self.max_length,
                                padding="post",
                                truncating="post")

        # Add to tensor
        train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))

        N_BATCH = train_x.shape[0] // self.BATCH_SIZE

        # Training
        if self.mode_training.lower() == "attention":
            self.training_with_attention(train_ds, N_BATCH)
        else:
            self.training(train_ds, N_BATCH)


if __name__ == "__main__":
    parser = ArgumentParser()

    # FIXME
    # Arguments users used when running command lines
    parser.add_argument("--inp-lang", required=True, type=str)
    parser.add_argument("--tar-lang", required=True, type=str)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--embedding-size", default=64, type=int)
    parser.add_argument("--hidden-units", default=256, type=int)
    parser.add_argument("--min-sentence", default=10, type=int)
    parser.add_argument("--max-sentence", default=14, type=int)
    parser.add_argument("--test-split-size", default=0.01, type=float)
    parser.add_argument("--mode-training", default="not_attention", type=str)
    parser.add_argument("--debug", default=False, type=bool)

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
    # SequenceToSequence("dataset/train.en.txt", "dataset/train.vi.txt").run()
    SequenceToSequence(inp_lang_path=args.inp_lang,
                       tar_lang_path=args.tar_lang,
                       batch_size=args.batch_size,
                       embedding_size=args.embedding_size,
                       hidden_units=args.hidden_units,
                       test_split_size=args.test_split_size,
                       epochs=args.epochs,
                       min_sentence=args.min_sentence,
                       max_sentence=args.max_sentence,
                       mode_training=args.mode_training,
                       debug=args.debug).run()

    # python train.py --inp-lang="dataset/train.en.txt" --tar-lang="dataset/train.vi.txt" --hidden-units=256 --embedding-size=128 --epochs=200 --test-split-size=0.01 --mode-training="attention" --debug=True
