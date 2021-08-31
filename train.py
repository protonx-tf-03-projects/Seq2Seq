import os
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from data import DatasetLoader
from argparse import ArgumentParser
from constant import MaskedSoftmaxCELoss, Bleu_score, CustomSchedule
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model.tests.model import Seq2SeqEncode, Seq2SeqDecode, BahdanauSeq2SeqDecode, LuongSeq2SeqDecoder


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
                 warmup_steps=80,
                 train_mode="attention",
                 attention_mode="luong",  # Bahdanau
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

        path_save = os.getcwd() + "/saved_models/"
        if not os.path.exists(path_save):
            os.mkdir(path_save)

        self.save_encoder = path_save + "/encoder.h5"
        self.save_decoder = path_save + "/decoder.h5"
        self.debug = debug

        # Load dataset
        self.inp_tensor, self.tar_tensor, self.inp_lang, self.tar_lang = DatasetLoader(self.inp_lang_path,
                                                                                       self.tar_lang_path,
                                                                                       self.min_sentence,
                                                                                       self.max_sentence).build_dataset()
        # Initialize optimizer
        learning_rate = CustomSchedule(self.hidden_units, warmup_steps=warmup_steps)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        # Initialize loss function
        self.loss = MaskedSoftmaxCELoss()

        # Initialize Bleu function
        self.bleu = Bleu_score()

        # Initialize encoder
        self.encoder = Seq2SeqEncode(self.inp_lang.vocab_size,
                                     self.embedding_size,
                                     self.hidden_units)

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
        tmp = 0
        for epoch in range(self.EPOCHS):
            loss = 0
            for batch_size, (x, y) in tqdm(enumerate(train_ds.batch(self.BATCH_SIZE).take(N_BATCH)), total=N_BATCH):
                with tf.GradientTape() as tape:
                    encoder_outs, last_state = self.encoder(x)
                    sos = tf.reshape(tf.constant([self.tar_lang.word2id['<sos>']] * self.BATCH_SIZE), shape=(-1, 1))
                    dec_input = tf.concat([sos, y[:, :-1]], 1)  # Teacher forcing
                    decode_out, _ = self.decoder(dec_input, last_state)
                    loss += self.loss(y, decode_out)

                train_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
                grads = tape.gradient(loss, train_vars)
                self.optimizer.apply_gradients(zip(grads, train_vars))

            bleu_score = self.evaluation(train_ds, self.debug)
            print("\n=================================================================")
            print(f'Epoch {epoch + 1} -- Loss: {loss} -- Bleu_score: {bleu_score}')
            print("=================================================================\n")
            if bleu_score > tmp:
                self.encoder.save_weights(self.save_encoder)
                self.decoder.save_weights(self.save_decoder)
                tmp = bleu_score

    def training_with_attention(self, train_ds, N_BATCH):
        tmp = 0
        for epoch in range(self.EPOCHS):
            total_loss = 0
            for batch_size, (x, y) in tqdm(enumerate(train_ds.batch(self.BATCH_SIZE).take(N_BATCH)), total=N_BATCH):
                loss = 0
                with tf.GradientTape() as tape:
                    encoder_outs, last_state = self.encoder(x)
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
            print(f'Epoch {epoch + 1} -- Loss: {total_loss} -- Bleu_score: {bleu_score}')
            print("=================================================================\n")
            if bleu_score > tmp:
                self.encoder.save_weights(self.save_encoder)
                self.decoder_attention.save_weights(self.save_decoder)
                tmp = bleu_score

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
            test_x = tf.expand_dims(test_, axis=0)
            _, last_state = self.encoder(test_x, training=False)

            input_decode = tf.reshape(tf.constant([self.tar_lang.word2id['<sos>']]), shape=(-1, 1))
            sentence = []
            for _ in range(len(test_y)):
                output, last_state = self.decoder(input_decode, last_state, training=False)
                output = tf.argmax(output, axis=2).numpy()
                input_decode = output
                sentence.append(output[0][0])
            score += self.bleu(self.tar_lang.vector_to_sentence(sentence),
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
            test_x = tf.expand_dims(test_, axis=0)
            encode_outs, last_state = self.encoder(test_x, training=False)

            input_decode = tf.constant([self.tar_lang.word2id['<sos>']])
            sentence = []
            for _ in range(len(test_y)):
                output, last_state = self.decoder_attention(input_decode, encode_outs, last_state, training=False)
                pred_id = tf.argmax(output, axis=1).numpy()
                input_decode = pred_id
                sentence.append(pred_id[0])

            score += self.bleu(self.tar_lang.vector_to_sentence(sentence),
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
    parser.add_argument("--warmup-steps", default=80, type=int)
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
                       batch_size=args.batch_size,
                       embedding_size=args.embedding_size,
                       hidden_units=args.hidden_units,
                       test_split_size=args.test_split_size,
                       epochs=args.epochs,
                       min_sentence=args.min_sentence,
                       max_sentence=args.max_sentence,
                       warmup_steps=args.warmup_steps,
                       train_mode=args.train_mode,
                       attention_mode=args.attention_mode,
                       debug=args.debug).run()

    # python train.py --inp-lang="dataset/train.en.txt" --tar-lang="dataset/train.vi.txt" --hidden-units=256 --embedding-size=128 --epochs=200 --test-split-size=0.01 --train-mode="attention" --debug=True
