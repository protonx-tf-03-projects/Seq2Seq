import os
import tensorflow as tf
from tqdm import tqdm
from data import DatasetLoader
from argparse import ArgumentParser
from constant import MaskedSoftmaxCELoss, Bleu_score, CustomSchedule, evaluation, evaluation_with_attention
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model.tests.model import LuongDecoder, BahdanauDecode, Decode, Encode


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

        # Initialize Bleu function
        self.bleu = Bleu_score()

        # Initialize Seq2Seq model
        self.input_vocab_size = len(self.inp_builder.word_index) + 1
        self.target_vocab_size = len(self.tar_builder.word_index) + 1

        # Initialize encoder
        self.encoder = Encode(self.input_vocab_size,
                              embedding_size,
                              hidden_units)

        # Initialize decoder with attention
        if self.train_mode.lower() == "attention":
            if attention_mode.lower() == "luong":
                self.decoder = LuongDecoder(self.target_vocab_size,
                                            embedding_size,
                                            hidden_units)
            else:
                self.decoder = BahdanauDecode(self.target_vocab_size,
                                              embedding_size,
                                              hidden_units)
        else:
            # Initialize decoder
            self.decoder = Decode(self.target_vocab_size,
                                  embedding_size,
                                  hidden_units)

        # Initialize translation
        self.checkpoint_prefix = os.path.join(self.path_save, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                              encoder=self.encoder,
                                              decoder=self.decoder)
        if retrain:
            print("[INFO] Retrain model with latest checkpoint...")
            self.checkpoint.restore(tf.train.latest_checkpoint(self.path_save)).expect_partial()

    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            # Teacher forcing
            sos = tf.reshape(tf.constant([self.tar_builder.word_index['<sos>']] * self.BATCH_SIZE),
                             shape=(-1, 1))
            dec_input = tf.concat([sos, y[:, :-1]], 1)
            # Encoder
            _, last_state = self.encoder(x)
            # Decoder
            outs, last_state = self.decoder(dec_input, last_state)
            # Loss
            loss = MaskedSoftmaxCELoss(y, outs)

        train_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
        grads = tape.gradient(loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))
        return loss

    def train_step_with_attention(self, x, y):
        loss = 0
        with tf.GradientTape() as tape:
            # Teaching force
            dec_input = tf.constant([self.tar_builder.word_index['<sos>']] * self.BATCH_SIZE)
            # Encoder
            encoder_outs, last_state = self.encoder(x)
            for i in range(1, y.shape[1]):
                # Decoder
                decode_out, last_state = self.decoder(dec_input, encoder_outs, last_state)
                # Loss
                loss += MaskedSoftmaxCELoss(y[:, i], decode_out)
                # Decoder input
                dec_input = y[:, i]

        train_vars = self.encoder.trainable_variables + self.decoder.trainable_variables
        grads = tape.gradient(loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))
        return loss

    def training(self):
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

        tmp = 0
        for epoch in range(self.EPOCHS):
            total_loss = 0
            for _, (x, y) in tqdm(enumerate(train_ds.batch(self.BATCH_SIZE).take(N_BATCH)), total=N_BATCH):
                if self.train_mode.lower() == "attention":
                    total_loss += self.train_step_with_attention(x, y)
                elif self.train_mode.lower() != "attention":
                    total_loss += self.train_step(x, y)

            if self.use_bleu:
                print("\n=================================================================")
                if self.train_mode.lower() == "attention":
                    bleu_score = evaluation_with_attention(encoder=self.encoder,
                                                           decoder=self.decoder,
                                                           test_ds=train_ds,
                                                           val_function=self.bleu,
                                                           inp_builder=self.inp_builder,
                                                           tar_builder=self.tar_builder,
                                                           test_split_size=self.test_split_size,
                                                           debug=self.debug)
                elif self.train_mode.lower() != "attention":
                    bleu_score = evaluation(encoder=self.encoder,
                                            decoder=self.decoder,
                                            test_ds=train_ds,
                                            val_function=self.bleu,
                                            inp_builder=self.inp_builder,
                                            tar_builder=self.tar_builder,
                                            test_split_size=self.test_split_size,
                                            debug=self.debug)

                print("-----------------------------------------------------------------")
                print(f'Epoch {epoch + 1} -- Loss: {total_loss} -- Bleu_score: {bleu_score}')
                if bleu_score > tmp:
                    self.checkpoint.save(file_prefix=self.checkpoint_prefix)
                    print("[INFO] Saved model in '{}' direction!".format(self.path_save))
                    tmp = bleu_score
                print("=================================================================\n")
            else:
                print("=================================================================")
                print(f'Epoch {epoch + 1} -- Loss: {total_loss}')
                print("=================================================================\n")

        print("[INFO] Saved model in '{}' direction!".format(self.path_save))
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)


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
    parser.add_argument("--bleu", default=False, type=bool)
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
            use_bleu=args.bleu,
            debug=args.debug).training()

    # python train.py --inp-lang="dataset/train.en.txt" --tar-lang="dataset/train.vi.txt" --hidden-units=256 --embedding-size=128 --epochs=200 --test-split-size=0.01 --train-mode="attention" --debug=True
