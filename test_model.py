import numpy as np
from data import DatasetLoader
from model.tests.model import Seq2SeqEncode, Seq2SeqDecode, LuongSeq2SeqDecoder, BahdanauSeq2SeqDecode
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import tensorflow as tf


def evaluation_with_attention(encoder, decoder_attention, test_ds):
    """
    :param test_ds: (inp_vocab, tar_vocab)
    :param (inp_lang, tar_lang)
    :return:
    """
    # Preprocessing testing data
    for test_, test_y in test_ds.shuffle(buffer_size=1, seed=1):
        test_x = tf.expand_dims(test_, axis=0)
        first_state = encoder.init_hidden_state(batch_size=1)
        encode_outs, last_state = encoder(test_x, first_state, training=False)

        input_decode = tf.constant([tar_lang.word2id['<sos>']])
        sentence = []
        for _ in range(len(test_y)):
            output, last_state = decoder_attention(input_decode, encode_outs, last_state, training=False)
            pred_id = tf.argmax(output, axis=1).numpy()
            input_decode = pred_id
            sentence.append(pred_id[0])
            print("-----------------------------------------------------------------")
            print("Input    : ", inp_lang.vector_to_sentence(test_.numpy()))
            print("Predicted: ", tar_lang.vector_to_sentence(sentence))
            print("Target   : ", tar_lang.vector_to_sentence(test_y.numpy()))
            print("=================================================================")


if __name__ == '__main__':
    raw_vi, raw_en, inp_lang, tar_lang = DatasetLoader("dataset/train.vi.txt", "dataset/train.en.txt").build_dataset()

    padded_sequences_vi = pad_sequences(raw_vi, maxlen=64, padding="post", truncating="post")
    padded_sequences_en = pad_sequences(raw_en, maxlen=64, padding="post", truncating="post")

    train_x, test_x, train_y, test_y = train_test_split(padded_sequences_vi, padded_sequences_en, test_size=0.1)

    train_x = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    test_x = tf.data.Dataset.from_tensor_slices((test_x, test_y))

    # for i, j in train_x.take(10):
    #     print("\n===================================================")
    #     print("Input: ", inp_lang.vector_to_sentence(i.numpy()))
    #     print("Target: ", tar_lang.vector_to_sentence(j.numpy()))

    train_x = train_x.batch(32)
    tmp_x, tmp_y = next(iter(train_x))

    # print("tmp_x", tmp_x.shape)

    embedding_size = 256
    hidden_unit = 1024
    BATCH_SIZE = 1

    encoder = Seq2SeqEncode(inp_lang.vocab_size, embedding_size, hidden_unit)
    encoder.load_weights("./save/encoder.h5")
    decoder = LuongSeq2SeqDecoder(tar_lang.vocab_size, embedding_size, hidden_unit)
    decoder.load_weights("./save/decoder.h5")
    evaluation_with_attention(encoder, decoder, test_x)
