import os

import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text
from argparse import ArgumentParser
from model.tests.model import SequenceToSequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers.experimental import preprocessing
from data import DatasetLoader


def tf_lower_and_split_punct(text):
    # Split accecented characters.
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)
    # Keep space, a to z, and select punctuation.
    text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
    # Add spaces around punctuation.
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
    # Strip whitespace.
    text = tf.strings.strip(text)

    text = tf.strings.join(['<sos>', text, '<eos>'], separator=' ')
    return text


class Translate(tf.Module):
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

        self.input_text_processor = preprocessing.TextVectorization(
            standardize=tf_lower_and_split_punct,
            max_tokens=max_length)

    def translate(self, input_text):
        vector = self.input_text_processor(input_text)
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

    def translate_with_attention(self, input_text):
        vector = self.input_text_processor(input_text)
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
                 hidden_units=256,
                 min_sentence=10,
                 max_sentence=14,
                 train_mode="not_attention",
                 attention_mode="luong"):
        self.weights_path = os.getcwd() + "/saved_models/"

        self.dataload = DatasetLoader(inp_lang_path,
                                      tar_lang_path,
                                      min_sentence,
                                      max_sentence)
        self.max_sentence = max_sentence
        self.inp_tensor, self.tar_tensor, self.inp_builder, self.tar_builder = self.dataload.build_dataset()

        inp_vocab_size = len(self.inp_builder.index_word) + 1
        tar_vocab_size = len(self.tar_builder.index_word) + 1

        self.model = SequenceToSequence(inp_vocab_size,
                                        tar_vocab_size,
                                        embedding_size,
                                        hidden_units,
                                        train_mode,
                                        attention_mode)
        latest = tf.train.latest_checkpoint(self.weights_path)

        self.model.load_weights(latest)

    def predict(self, sentence):
        """
        :param test_ds: (inp_vocab, tar_vocab)
        :param (inp_lang, tar_lang)
        :return:
        """
        sentence = self.dataload.remove_punctuation(sentence)
        if args.min_sentence < len(sentence.split()) < args.max_sentence:
            vector = self.inp_builder.sequences_to_texts([sentence.split()])
            test_x = pad_sequences(vector,
                                   maxlen=self.max_sentence,
                                   padding="post",
                                   truncating="post")
            _, last_state = self.model.encoder(test_x)

            input_decode = tf.reshape(tf.constant([self.tar_builder.word_index['<sos>']]), shape=(-1, 1))
            pred_sentence = ""
            for _ in range(self.max_sentence):
                output, last_state = self.model.decoder(input_decode, last_state)
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
        sentence = self.dataload.remove_punctuation(sentence)
        if args.min_sentence < len(sentence.split()) < args.max_sentence:

            vector = self.inp_builder.sequences_to_texts(sentence)
            test_x = tf.expand_dims(vector, axis=0)
            encode_outs, last_state = self.model.encoder(test_x)
            input_decode = tf.constant([self.tar_builder.word_index['<sos>']])
            vector = []
            for _ in range(self.max_sentence):
                output, last_state = self.model.decoder(input_decode, encode_outs, last_state)
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
    with open(args.test_path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            if args.attention_mode.lower() == "attention":
                define.predict_with_attention(line)
            else:
                define.predict(line)

    # python predict.py --test-path="dataset/train.en.txt" --inp-lang-path="dataset/train.en.txt" --tar-lang-path="dataset/train.vi.txt"
