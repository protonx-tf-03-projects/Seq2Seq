import os
import tensorflow as tf
from argparse import ArgumentParser
from model.tests.model import LuongDecoder, Encode, BahdanauDecode, Decode
from tensorflow.keras.preprocessing.sequence import pad_sequences
from data import DatasetLoader


class Translate(tf.keras.Model):
    def __init__(self,
                 inp_lang_path,
                 tar_lang_path,
                 encoder,
                 decoder,
                 tar_builder,
                 min_length=10,
                 max_length=14):
        super(Translate, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.tar_builder = tar_builder
        self.max_length = max_length

        self.loader = DatasetLoader(inp_lang_path,
                                    tar_lang_path,
                                    min_length,
                                    max_length)
        _, _, self.inp_builder, self.tar_builder = self.loader.build_dataset()

    def translate_enroll(self, input_text):
        vector = self.loader.remove_punctuation(input_text)
        # Encoder
        _, last_state = self.encoder(vector)
        # Process decoder input
        input_decode = tf.reshape(tf.constant([self.tar_builder.word_index['<sos>']]), shape=(-1, 1))
        pred_sentence = ""
        for _ in range(self.max_length):
            output, last_state = self.decoder(input_decode, last_state)
            pred_id = tf.argmax(output, axis=2).numpy()
            input_decode = pred_id
            pred_sentence += " " + self.tar_builder.index_word[pred_id[0][0]]
        text = [w for w in pred_sentence.split() if w not in ["<sos>", "<eos>"]]
        return text

    def translate_with_attention_enroll(self, input_text):
        vector = self.loader.remove_punctuation(input_text)
        test_x = tf.expand_dims(vector, axis=0)
        # Encoder
        encode_outs, last_state = self.model.encoder(test_x)
        # Process decoder input
        input_decode = tf.constant([self.tar_builder.word_index['<sos>']])
        pred_sentence = ""
        for _ in range(self.max_sentence):
            output, last_state = self.model.decoder(input_decode, encode_outs, last_state)
            pred_id = tf.argmax(output, axis=1).numpy()
            input_decode = pred_id
            pred_sentence += " " + self.tar_builder.index_word[pred_id[0]]
        text = [w for w in pred_sentence.split() if w not in ["<sos>", "<eos>"]]
        return text


class PredictionSentence:
    def __init__(self,
                 inp_lang_path,
                 tar_lang_path,
                 embedding_size=64,
                 hidden_units=128,
                 min_sentence=10,
                 max_sentence=14,
                 learning_rate=0.005,
                 train_mode="not_attention",
                 attention_mode="luong"):

        self.path_save = os.getcwd() + "/saved_models"
        if not os.path.exists(self.path_save):
            os.mkdir(self.path_save)

        self.loader = DatasetLoader(inp_lang_path,
                                    tar_lang_path,
                                    min_sentence,
                                    max_sentence)
        self.max_sentence = max_sentence
        self.inp_tensor, self.tar_tensor, self.inp_builder, self.tar_builder = self.loader.build_dataset()

        # Initialize Seq2Seq model
        self.input_vocab_size = len(self.inp_builder.word_index) + 1
        self.target_vocab_size = len(self.tar_builder.word_index) + 1

        # Initialize optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

        # Initialize encoder
        self.encoder = Encode(self.input_vocab_size,
                              embedding_size,
                              hidden_units)

        # Initialize decoder with attention
        if train_mode.lower() == "attention":
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

        self.checkpoint.restore(tf.train.latest_checkpoint(self.path_save)).expect_partial()

    def __preprocess_text__(self, text):
        input_text = self.loader.remove_punctuation(text)
        vector = self.inp_builder.texts_to_sequences([input_text.split()])
        text = pad_sequences(vector,
                             maxlen=self.max_sentence,
                             padding="post",
                             truncating="post")
        return text

    def translate_enroll(self, input_text):
        vector = self.__preprocess_text__(input_text)
        # Encoder
        _, last_state = self.encoder(vector)
        # Process decoder input
        input_decode = tf.reshape(tf.constant([self.tar_builder.word_index['<sos>']]), shape=(-1, 1))
        pred_sentence = ""
        for _ in range(self.max_sentence):
            output, last_state = self.decoder(input_decode, last_state)
            pred_id = tf.argmax(output, axis=2).numpy()
            input_decode = pred_id
            pred_sentence += " " + self.tar_builder.index_word[pred_id[0][0]]
        text = [w for w in pred_sentence.split() if w not in ["<sos>", "<eos>"]]
        print("-----------------------------------------------------------------")
        print("Input:   ", input_text)
        print("Predicted: ", text)
        print("-----------------------------------------------------------------")

    def translate_with_attention_enroll(self, input_text):
        vector = self.__preprocess_text__(input_text)
        test_x = tf.expand_dims(vector, axis=0)
        # Encoder
        encode_outs, last_state = self.encoder(test_x)
        # Process decoder input
        input_decode = tf.constant([self.tar_builder.word_index['<sos>']])
        pred_sentence = ""
        for _ in range(self.max_sentence):
            output, last_state = self.decoder(input_decode, encode_outs, last_state)
            pred_id = tf.argmax(output, axis=1).numpy()
            input_decode = pred_id
            pred_sentence += " " + self.tar_builder.index_word[pred_id[0]]
        text = [w for w in pred_sentence.split() if w not in ["<sos>", "<eos>"]]
        return text

    def predict(self, sentence):
        """
        :param test_ds: (inp_vocab, tar_vocab)
        :param (inp_lang, tar_lang)
        :return:
        """
        sentence = self.loader.remove_punctuation(sentence)
        if args.min_sentence <= len(sentence.split()) <= args.max_sentence:
            vector = self.__preprocess_text__(sentence)
            _, last_state = self.encoder(vector)

            input_decode = tf.reshape(tf.constant([self.tar_builder.word_index['<sos>']]), shape=(-1, 1))
            pred_sentence = ""
            for _ in range(self.max_sentence):
                output, last_state = self.decoder(input_decode, last_state)
                pred_id = tf.argmax(output, axis=2).numpy()
                input_decode = pred_id
                pred_sentence += " " + self.tar_builder.index_word[pred_id[0][0]]
            print("-----------------------------------------------------------------")
            print("Input:   ", sentence)
            print("Predicted: ", pred_sentence)
            print("-----------------------------------------------------------------")

    def predict_with_attention(self, sentence):
        """
        :param test_ds: (inp_vocab, tar_vocab)
        :param (inp_lang, tar_lang)
        :return:
        """
        sentence = self.loader.remove_punctuation(sentence)
        if args.min_sentence <= len(sentence.split()) <= args.max_sentence:
            vector = self.__preprocess_text__(sentence)
            test_x = tf.expand_dims(vector, axis=0)
            encode_outs, last_state = self.encoder(test_x)
            input_decode = tf.constant([self.tar_builder.word_index['<sos>']])
            vector = []
            for _ in range(self.max_sentence):
                output, last_state = self.decoder(input_decode, encode_outs, last_state)
                pred_id = tf.argmax(output, axis=1).numpy()
                input_decode = pred_id
                vector.append(pred_id[0])
            print("-----------------------------------------------------------------")
            print("Input:   ", sentence)
            print("Predicted: ", self.tar_builder.vector_to_sentence(vector))
            print("-----------------------------------------------------------------")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--test-path", required=True, type=str)
    parser.add_argument("--inp-lang-path", required=True, type=str)
    parser.add_argument("--tar-lang-path", required=True, type=str)
    parser.add_argument("--embedding-size", default=64, type=str)
    parser.add_argument("--hidden-units", default=128, type=str)
    parser.add_argument("--min-sentence", default=10, type=str)
    parser.add_argument("--max-sentence", default=14, type=str)
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
    define = PredictionSentence(inp_lang_path=args.inp_lang_path,
                                tar_lang_path=args.tar_lang_path,
                                hidden_units=args.hidden_units,
                                embedding_size=args.embedding_size,
                                min_sentence=args.min_sentence,
                                max_sentence=args.max_sentence,
                                train_mode=args.train_mode,
                                attention_mode=args.attention_mode)
    if args.predict_a_sentence:
        sentence = input("Enter the sentence to translate: ")
        if args.attention_mode.lower() == "attention":
            define.translate_with_attention_enroll(sentence)
        else:
            define.translate_enroll(sentence)
    else:
        with open(args.test_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                if args.attention_mode.lower() == "attention":
                    define.predict_with_attention(line)
                else:
                    define.predict(line)

    # python predict.py --test-path="dataset/train.en.txt" --inp-lang-path="dataset/train.en.txt" --tar-lang-path="dataset/train.vi.txt"
