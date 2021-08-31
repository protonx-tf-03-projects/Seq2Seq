import numpy as np
from data import DatasetLoader
from model.tests.model import SequenceToSequence
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import tensorflow as tf


def predict(model, test_ds):
    """
    :param test_ds: (inp_vocab, tar_vocab)
    :param (inp_lang, tar_lang)
    :return:
    """
    # Preprocessing testing data
    for test_, test_y in test_ds.shuffle(buffer_size=1, seed=1).take(1):
        test_x = tf.expand_dims(test_, axis=0)
        _, last_state = model.encoder(test_x)

        input_decode = tf.reshape(tf.constant([tar_lang.word2id['<sos>']]), shape=(-1, 1))
        sentence = []
        for _ in range(len(test_y)):
            output, last_state = model.decoder(input_decode, last_state, training=False)
            output = tf.argmax(output, axis=2).numpy()
            input_decode = output
            sentence.append(output[0][0])
            print("-----------------------------------------------------------------")
            print("Input    : ", inp_lang.vector_to_sentence(test_.numpy()))
            print("Predicted: ", tar_lang.vector_to_sentence(sentence))
            print("Target   : ", tar_lang.vector_to_sentence(test_y.numpy()))
            print("=================================================================")


if __name__ == '__main__':
    raw_vi, raw_en, inp_lang, tar_lang = DatasetLoader("dataset/train.en.txt", "dataset/train.vi.txt").build_dataset()

    padded_sequences_vi = pad_sequences(raw_vi, maxlen=14, padding="post", truncating="post")
    padded_sequences_en = pad_sequences(raw_en, maxlen=14, padding="post", truncating="post")

    train_ds = tf.data.Dataset.from_tensor_slices((padded_sequences_vi, padded_sequences_en))

    embedding_size = 32
    hidden_unit = 64
    BATCH_SIZE = 1

    model = SequenceToSequence(inp_lang.vocab_size,
                               tar_lang.vocab_size,
                               embedding_size,
                               hidden_unit,
                               "not_attention")
    model.load_weights("saved_models/")
    predict(model, train_ds)
