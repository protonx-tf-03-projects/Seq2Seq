import os
import tensorflow as tf

from argparse import ArgumentParser
from model.tests.model import SequenceToSequence

from data import DatasetLoader


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

        self.inp_tensor, self.tar_tensor, self.inp_builder, self.tar_builder = self.dataload.build_dataset()

        self.model = SequenceToSequence(self.inp_builder.vocab_size,
                                        self.tar_builder.vocab_size,
                                        embedding_size,
                                        hidden_units,
                                        train_mode,
                                        attention_mode)
        self.model.load_weights(self.weights_path)

    def predict(self, sentence):
        """
        :param test_ds: (inp_vocab, tar_vocab)
        :param (inp_lang, tar_lang)
        :return:
        """
        sentence = self.dataload.remove_punctuation_digits(sentence)
        vector = self.inp_builder.sentence_to_vector(sentence)
        test_x = tf.expand_dims(vector, axis=0)
        _, last_state = self.model.encoder(test_x)

        input_decode = tf.reshape(tf.constant([self.tar_builder.word2id['<sos>']]), shape=(-1, 1))
        sentence = []
        for _ in range(self.tar_builder.max_len):
            output, last_state = self.model.decoder(input_decode, last_state)
            output = tf.argmax(output, axis=2).numpy()
            input_decode = output
            sentence.append(output[0][0])
        print("-----------------------------------------------------------------")
        print("Predicted: ", self.tar_builder.vector_to_sentence(sentence))
        print("-----------------------------------------------------------------")

    def predict_with_attention(self, sentence):
        """
        :param test_ds: (inp_vocab, tar_vocab)
        :param (inp_lang, tar_lang)
        :return:
        """
        sentence = self.dataload.remove_punctuation_digits(sentence)
        vector = self.inp_builder.sentence_to_vector(sentence)
        test_x = tf.expand_dims(vector, axis=0)
        encode_outs, last_state = self.model.encoder(test_x)
        input_decode = tf.constant([self.tar_builder.word2id['<sos>']])
        sentence = []
        for _ in range(self.tar_builder.max_len):
            output, last_state = self.model.decoder(input_decode, encode_outs, last_state)
            pred_id = tf.argmax(output, axis=1).numpy()
            input_decode = pred_id
            sentence.append(pred_id[0])
        print("-----------------------------------------------------------------")
        print("Predicted: ", self.tar_builder.vector_to_sentence(sentence))
        print("-----------------------------------------------------------------")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--test-path", required=True, type=str)
    parser.add_argument("--inp-lang-path", required=True, type=str)
    parser.add_argument("--tar-lang-path", required=True, type=str)
    parser.add_argument("--embedding-size", default=64, type=str)
    parser.add_argument("--hidden-units", default=256, type=str)
    parser.add_argument("--min-sentence", default=10, type=str)
    parser.add_argument("--max-sentence", default=14, type=str)
    parser.add_argument("--attention-mode", default="not_attention", type=str)
    parser.add_argument("--train-mode", default="luong", type=str)

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
            if args.attention_mode.lower() == "not_attention":
                print(line)
                define.predict(line)
            elif args.attention_mode.lower() == "attention":
                print(line)
                define.predict_with_attention(line)
            else:
                print(EOFError)

    # python predict.py --test-path="dataset/train.en.txt" --inp-lang-path="dataset/train.en.txt" --tar-lang-path="dataset/train.vi.txt"