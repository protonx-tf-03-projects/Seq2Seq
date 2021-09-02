import os
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from data import DatasetLoader
from argparse import ArgumentParser
from constant import MaskedSoftmaxCELoss, Bleu_score, CustomSchedule, evaluation, evaluation_with_attention
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model.tests.model import SequenceToSequence


class Seq2Seq:
    def __init__(self,
                 inp_lang_path,
                 tar_lang_path,
                 embedding_size=64,
                 hidden_units=256,
                 learning_rate=0.001,
                 test_split_size=0.005,
                 epochs=400,
                 batch_size=128,
                 min_sentence=10,
                 max_sentence=14,
                 warmup_steps=80,
                 train_mode="attention",
                 attention_mode="luong",  # Bahdanau
                 use_lr_schedule=False,
                 use_bleu=False,
                 retrain=False,
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
        self.train_mode = train_mode
        self.attention_mode = attention_mode

        self.path_save = os.getcwd() + "/saved_models"
        if not os.path.exists(self.path_save):
            os.mkdir(self.path_save)

        self.save_checkpoint = self.path_save + "/model.ckpt"
        self.debug = debug
        self.use_bleu = use_bleu

        # Load dataset
        self.inp_tensor, self.tar_tensor, self.inp_builder, self.tar_builder = DatasetLoader(self.inp_lang_path,
                                                                                             self.tar_lang_path,
                                                                                             self.min_sentence,
                                                                                             self.max_sentence).build_dataset()
        # Initialize optimizer
        if use_lr_schedule:
            learning_rate = CustomSchedule(self.hidden_units, warmup_steps=warmup_steps)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        # Initialize loss function
        self.loss = MaskedSoftmaxCELoss()

        # Initialize Bleu function
        self.bleu = Bleu_score()

        # Initialize Seq2Seq model
        if retrain and os.listdir(self.path_save) != []:
            print("[INFO] Start retrain...")
            self.model = SequenceToSequence(self.inp_builder.vocab_size,
                                            self.tar_builder.vocab_size,
                                            self.embedding_size,
                                            self.hidden_units,
                                            self.train_mode,
                                            self.attention_mode)
            self.model.load_weights(self.path_save)
        else:
            self.model = SequenceToSequence(self.inp_builder.vocab_size,
                                            self.tar_builder.vocab_size,
                                            self.embedding_size,
                                            self.hidden_units,
                                            self.train_mode,
                                            self.attention_mode)

    def training(self, train_ds, N_BATCH):
        tmp = 0
        for epoch in range(self.EPOCHS):
            loss = 0
            for batch_size, (x, y) in tqdm(enumerate(train_ds.batch(self.BATCH_SIZE).take(N_BATCH)), total=N_BATCH):
                with tf.GradientTape() as tape:
                    sos = tf.reshape(tf.constant([self.tar_builder.word2id['<sos>']] * self.BATCH_SIZE), shape=(-1, 1))
                    dec_input = tf.concat([sos, y[:, :-1]], 1)  # Teacher forcing
                    outs = self.model(x, dec_input)
                    loss += self.loss(y, outs)

                train_vars = self.model.trainable_variables
                grads = tape.gradient(loss, train_vars)
                self.optimizer.apply_gradients(zip(grads, train_vars))

                print("\n=================================================================")
                if self.use_bleu:
                    bleu_score = evaluation_with_attention(model=self.model,
                                                           test_ds=train_ds,
                                                           val_function=self.bleu,
                                                           inp_builder=self.inp_builder,
                                                           tar_builder=self.tar_builder,
                                                           test_split_size=self.test_split_size,
                                                           debug=self.debug)
                    print(f'Epoch {epoch + 1} -- Loss: {loss} -- Bleu_score: {bleu_score}')
                    if bleu_score > tmp:
                        self.model.save_weights(self.save_checkpoint)
                        print("[INFO] Saved model in '{}' direction!".format(self.path_save))
                        tmp = bleu_score
                else:
                    print(f'Epoch {epoch + 1} -- Loss: {loss}')
                print("=================================================================\n")
            self.model.save_weights(self.save_checkpoint)

    def training_with_attention(self, train_ds, N_BATCH):
        tmp = 0
        for epoch in range(self.EPOCHS):
            total_loss = 0
            for batch_size, (x, y) in tqdm(enumerate(train_ds.batch(self.BATCH_SIZE).take(N_BATCH)), total=N_BATCH):
                loss = 0
                with tf.GradientTape() as tape:
                    dec_input = tf.constant([self.tar_builder.word2id['<sos>']] * self.BATCH_SIZE)
                    for i in range(1, y.shape[1]):
                        decode_out = self.model(x, dec_input)
                        loss += self.loss(y[:, i], decode_out)
                        dec_input = y[:, i]

                train_vars = self.model.trainable_variables
                grads = tape.gradient(loss, train_vars)
                self.optimizer.apply_gradients(zip(grads, train_vars))
                total_loss += loss

            print("\n=================================================================")
            if self.use_bleu:
                bleu_score = evaluation_with_attention(model=self.model,
                                                       test_ds=train_ds,
                                                       val_function=self.bleu,
                                                       inp_builder=self.inp_builder,
                                                       tar_builder=self.tar_builder,
                                                       test_split_size=self.test_split_size,
                                                       debug=self.debug)
                print(f'Epoch {epoch + 1} -- Loss: {total_loss} -- Bleu_score: {bleu_score}')
                if bleu_score > tmp:
                    self.model.save_weights(self.save_checkpoint)
                    print("[INFO] Saved model in '{}' direction!".format(self.path_save))
                    tmp = bleu_score
            else:
                print(f'Epoch {epoch + 1} -- Loss: {total_loss}')
            print("=================================================================\n")
        self.model.save_weights(self.save_checkpoint)

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
        if self.train_mode.lower() == "attention":
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
    parser.add_argument("--learning-rate", default=0.005, type=float)
    parser.add_argument("--train-mode", default="not_attention", type=str)
    parser.add_argument("--attention-mode", default="luong", type=str)
    parser.add_argument("--use-lr-schedule", default=False, type=bool)
    parser.add_argument("--retrain", default=False, type=bool)
    parser.add_argument("--use-bleu", default=False, type=bool)
    parser.add_argument("--debug", default=False, type=bool)

    args = parser.parse_args()

    # FIXME
    # Project Description

    print('---------------------Welcome to Hợp tác xã kiên trì-------------------')
    print('Github: https://github.com/Xunino')
    print('Email: ndlinh.ai@gmail.com')
    print('------------------------------------------------------------------------')
    print(f'Training Sequences To Sequences model with hyper-params:')
    print('------------------------------------')
    for k, v in vars(args).items():
        print(f"|  +) {k} = {v}")
    print('====================================')

    # FIXME
    # Do Training
    Seq2Seq(inp_lang_path=args.inp_lang,
            tar_lang_path=args.tar_lang,
            batch_size=args.batch_size,
            embedding_size=args.embedding_size,
            hidden_units=args.hidden_units,
            test_split_size=args.test_split_size,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            min_sentence=args.min_sentence,
            max_sentence=args.max_sentence,
            warmup_steps=args.warmup_steps,
            train_mode=args.train_mode,
            attention_mode=args.attention_mode,
            use_lr_schedule=args.use_lr_schedule,
            retrain=args.retrain,
            use_bleu=args.use_bleu,
            debug=args.debug).run()

    # python train.py --inp-lang="dataset/train.en.txt" --tar-lang="dataset/train.vi.txt" --hidden-units=256 --embedding-size=128 --epochs=200 --test-split-size=0.01 --train-mode="attention" --debug=True
