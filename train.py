import os
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from data import DatasetLoader
from argparse import ArgumentParser
from constant import MaskedSoftmaxCELoss, Bleu_score
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model.tests.model import Seq2SeqEncode, Seq2SeqDecode, BahdanauSeq2SeqDecode, LuongSeq2SeqDecoder


class SequenceToSequence:
    def __init__(self,
                 inp_lang_path,
                 tar_lang_path,
                 learning_rate=0.001,
                 embedding_size=64,
                 hidden_units=256,
                 test_split_size=0.005,
                 epochs=400,
                 batch_size=128,
                 min_sentence=10,
                 max_sentence=14,
                 train_mode="attention",
                 attention_mode="luong",  # Bahdanau
                 save_encoder="save/weights/encoder.h5",
                 save_decoder="save/weights/decoder.h5",
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
        self.mode_training = train_mode
        self.attention_mode = attention_mode
        self.save_encoder = save_encoder
        self.save_decoder = save_decoder
        self.debug = debug

        # Load dataset
        self.inp_tensor, self.tar_tensor, self.inp_lang, self.tar_lang = DatasetLoader(self.inp_lang_path,
                                                                                       self.tar_lang_path,
                                                                                       self.min_sentence,
                                                                                       self.max_sentence).build_dataset()
        # Initialize optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

        # Initialize loss function
        self.loss = MaskedSoftmaxCELoss()

        # Initialize encoder
        self.encoder = Seq2SeqEncode(self.inp_lang.vocab_size,
                                     self.embedding_size,
                                     self.hidden_units)
        # Initialize first state
        self.first_state = self.encoder.init_hidden_state(self.BATCH_SIZE)

        # Initialize decoder with attention
        if self.mode_training.lower() == "attention":
            if self.attention_mode.lower() == "luong":
                self.decoder_attention = LuongSeq2SeqDecoder(self.tar_lang.vocab_size,
                                                             self.embedding_size,
                                                             self.hidden_units)
            else:
                self.decoder_attention = BahdanauSeq2SeqDecode(self.tar_lang.vocab_size,
                                                               self.embedding_size,
                                                               self.hidden_units)
        else:
            # Initialize decoder
            self.decoder = Seq2SeqDecode(self.tar_lang.vocab_size,
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
                    loss += self.loss(y, decode_out)

                train_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
                grads = tape.gradient(loss, train_vars)
                self.optimizer.apply_gradients(zip(grads, train_vars))

            bleu_score = self.evaluation(train_ds, self.debug)
            print("\n=================================================================")
            print(f'Epoch {epoch + 1} -- Loss: {loss} -- Bleu_score: {round(bleu_score, 2) * 100}')
            print("=================================================================\n")

        # self.encoder.save_weights(self.save_encoder)
        # self.decoder.save_weights(self.save_decoder)

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
                        loss += self.loss(y[:, i], decode_out)
                        dec_input = y[:, i]

                train_vars = self.encoder.trainable_variables + self.decoder_attention.trainable_variables
                grads = tape.gradient(loss, train_vars)
                self.optimizer.apply_gradients(zip(grads, train_vars))
                total_loss += loss

            bleu_score = self.evaluation_with_attention(train_ds, self.debug)
            print("\n=================================================================")
            print(f'Epoch {epoch + 1} -- Loss: {total_loss} -- Bleu_score: {round(bleu_score, 2) * 100}')
            print("=================================================================\n")

        # self.encoder.save_weights(self.save_encoder)
        # self.decoder_attention.save_weights(self.save_decoder)

    def evaluation(self, test_ds, debug=False):
        """
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
            encode_outs, last_state = self.encoder(test_x, first_state, training=False)

            input_decode = np.array([self.tar_lang.word2id['<sos>']])
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
    parser.add_argument("--learning-rate", default=0.01, type=float)
    parser.add_argument("--batch-size", default=128, type=int)
    parser.add_argument("--epochs", default=1000, type=int)
    parser.add_argument("--embedding-size", default=64, type=int)
    parser.add_argument("--hidden-units", default=256, type=int)
    parser.add_argument("--min-sentence", default=10, type=int)
    parser.add_argument("--max-sentence", default=14, type=int)
    parser.add_argument("--test-split-size", default=0.01, type=float)
    parser.add_argument("--train-mode", default="not_attention", type=str)
    parser.add_argument("--attention-mode", default="luong", type=str)
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
    SequenceToSequence(inp_lang_path=args.inp_lang,
                       tar_lang_path=args.tar_lang,
                       learning_rate=args.learning_rate,
                       batch_size=args.batch_size,
                       embedding_size=args.embedding_size,
                       hidden_units=args.hidden_units,
                       test_split_size=args.test_split_size,
                       epochs=args.epochs,
                       min_sentence=args.min_sentence,
                       max_sentence=args.max_sentence,
                       train_mode=args.train_mode,
                       attention_mode=args.attention_mode,
                       debug=args.debug).run()

    # python train.py --inp-lang="dataset/train.en.txt" --tar-lang="dataset/train.vi.txt" --hidden-units=256 --embedding-size=128 --epochs=200 --test-split-size=0.01 --train-mode="attention" --debug=True
