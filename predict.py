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
    text = "But now , we have a real technology to do this ."
    PredictionSentence("dataset/train.en.txt", "dataset/train.vi.txt").predict(text)

    # parser = ArgumentParser()
    # parser.add_argument("--batch-size", default=64, type=int)
    # parser.add_argument("--epochs", default=1000, type=int)
    #
    # # FIXME
    # args = parser.parse_args()
    #
    # # FIXME
    # # Project Description
    #
    # print('---------------------Welcome to ${name}-------------------')
    # print('Github: ${accout}')
    # print('Email: ${email}')
    # print('---------------------------------------------------------------------')
    # print('Training ${name} model with hyper-params:')  # FIXME
    # print('===========================')
    #
    # # FIXME
    # # Do Training
