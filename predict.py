import json
import os
import tensorflow as tf
from argparse import ArgumentParser
from data import remove_punctuation
from metrics import CustomSchedule
from tensorflow.keras.preprocessing.sequence import pad_sequences
from model.Encoder import Encode
from model.Decoder import Decode
from model.BahdanauDecode import BahdanauDecode
from model.LuongDecoder import LuongDecoder


class PredictionSentence:
    def __init__(self,
                 embedding_size=64,
                 hidden_units=128,
                 max_sentence=100,
                 learning_rate=0.001,
                 train_mode="not_attention",
                 attention_mode="luong"):

        home = os.getcwd()
        self.max_sentence = max_sentence
        self.save_dict = home + "/saved_models/{}_vocab.json"

        self.inp_builder = self.load_tokenizer(name_vocab="input")
        self.tar_builder = self.load_tokenizer(name_vocab="target")
        self.values = list(self.tar_builder.values())
        self.keys = list(self.tar_builder.keys())

        # Initialize Seq2Seq model
        input_vocab_size = len(self.inp_builder) + 1
        target_vocab_size = len(self.tar_builder) + 1

        # Initialize encoder
        self.encoder = Encode(input_vocab_size,
                              embedding_size,
                              hidden_units)

        # # Initialize decoder with attention
        if train_mode.lower() == "attention":
            if attention_mode.lower() == "luong":
                self.decoder = LuongDecoder(target_vocab_size,
                                            embedding_size,
                                            hidden_units)
            else:
                self.decoder = BahdanauDecode(target_vocab_size,
                                              embedding_size,
                                              hidden_units)
        else:
            # Initialize decoder
            self.decoder = Decode(target_vocab_size,
                                  embedding_size,
                                  hidden_units)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        # Initialize translation
        self.path_save = home + "/saved_models"
        self.checkpoint_prefix = os.path.join(self.path_save, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer,
                                              encoder=self.encoder,
                                              decoder=self.decoder)
        self.checkpoint.restore(tf.train.latest_checkpoint(self.path_save)).expect_partial()

    def __preprocess_input_text__(self, text):
        text = remove_punctuation(text)
        vector = [[self.inp_builder[w] for w in text.split() if w in list(self.inp_builder.keys())]]
        sentence = pad_sequences(vector,
                                 maxlen=self.max_sentence,
                                 padding="post",
                                 truncating="post")
        return sentence

    def load_tokenizer(self, name_vocab):
        f = open(self.save_dict.format(name_vocab), "r", encoding="utf-8")
        return json.load(f)

    def translate_enroll(self, input_text):
        vector = self.__preprocess_input_text__(input_text)
        # Encoder
        _, last_state = self.encoder(vector)
        # Process decoder input
        input_decode = tf.reshape(tf.constant([self.tar_builder['<sos>']]), shape=(-1, 1))
        pred_sentence = ""
        for _ in range(self.max_sentence):
            output, last_state = self.decoder(input_decode, last_state)
            pred_id = tf.argmax(output, axis=2).numpy()
            input_decode = pred_id
            word = self.keys[self.values.index(pred_id[0])]
            if word not in ["<sos>", "<eos>"]:
                pred_sentence += " " + word
            if word in ["<eos>"]:
                break
        print("-----------------------------------------------------------------")
        print("Input     : ", input_text)
        print("Translate :", pred_sentence)
        print("-----------------------------------------------------------------")

    def translate_with_attention_enroll(self, input_text):
        vector = self.__preprocess_input_text__(input_text)
        # Encoder
        encode_outs, last_state = self.encoder(vector)
        # Process decoder input
        input_decode = tf.constant([self.tar_builder['<sos>']])
        pred_sentence = ""
        for _ in range(self.max_sentence):
            output, last_state = self.decoder(input_decode, encode_outs, last_state)
            pred_id = tf.argmax(output, axis=1).numpy()
            input_decode = pred_id
            word = self.keys[self.values.index(pred_id[0])]
            if word not in ["<sos>", "<eos>"]:
                pred_sentence += " " + word
            if word in ["<eos>"]:
                break
        print("-----------------------------------------------------------------")
        print("Input     : ", input_text)
        print("Translate :", pred_sentence)
        print("-----------------------------------------------------------------")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--test-path", required=True, type=str)
    parser.add_argument("--embedding-size", default=64, type=int)
    parser.add_argument("--hidden-units", default=128, type=int)
    parser.add_argument("--max-sentence", default=100, type=int)
    parser.add_argument("--attention-mode", default="luong", type=str)
    parser.add_argument("--train-mode", default="not_attention", type=str)
    parser.add_argument("--predict-a-sentence", default=False, type=bool)

    args = parser.parse_args()

    print('---------------------Welcome to Hợp tác xã Kiên trì-------------------')
    print('Github: https://github.com/Xunino')
    print('Email: ndlinh.ai@gmail.com')
    print('---------------------------------------------------------------------')
    print('Predicting Sequence To Sequence model with hyper-params:')
    print('------------------------------------')
    for k, v in vars(args).items():
        print(f"|  +) {k} = {v}")
    print('====================================')

    # FIXME
    # Do Predict
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    define = PredictionSentence(hidden_units=args.hidden_units,
                                embedding_size=args.embedding_size,
                                max_sentence=args.max_sentence,
                                train_mode=args.train_mode,
                                attention_mode=args.attention_mode)
    if args.predict_a_sentence:
        print("----------------------------------------------------")
        input_text = input("[INFO] Enter the sentence to translate: ")
        if args.train_mode.lower() == "attention":
            define.translate_with_attention_enroll(input_text)
        else:
            define.translate_enroll(input_text)
    else:
        with open(args.test_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                if args.train_mode.lower() == "attention":
                    define.translate_with_attention_enroll(line)
                else:
                    define.translate_enroll(line)
    '''
    No attention
    # python predict.py --test-path="dataset/train.en.txt" --max-sentence=14
    
    With attention
    # python predict.py --test-path="dataset/train.en.txt" --hidden-units=512 --embedding-size=256 --max-sentence=20 --train-mode="attention"
    '''
