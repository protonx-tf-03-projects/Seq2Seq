import numpy as np
from data import DatasetLoader
from model.tests.model_test import Seq2SeqEncode, Seq2SeqDecode
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import tensorflow as tf

if __name__ == '__main__':
    raw_vi, raw_en, caches = DatasetLoader("dataset/train.vi.txt", "dataset/train.en.txt").build_dataset()

    padded_sequences_vi = pad_sequences(raw_vi, maxlen=64, padding="post", truncating="post")
    padded_sequences_en = pad_sequences(raw_en, maxlen=64, padding="post", truncating="post")

    train_x, test_x, train_y, test_y = train_test_split(padded_sequences_vi, padded_sequences_en, test_size=0.1)

    train_x = tf.data.Dataset.from_tensor_slices((train_x, train_y))
    test_x = tf.data.Dataset.from_tensor_slices((test_x, test_y))

    train_x = train_x.batch(32)
    tmp_x, tmp_y = next(iter(train_x))

    print("tmp_x", tmp_x.shape)

    embedding_size = 64
    vocab_size = 10000
    hidden_unit = 256
    BATCH_SIZE = 32

    encoder = Seq2SeqEncode(vocab_size, embedding_size, hidden_unit, n_layers=1)
    state = encoder._init_hidden_state_(BATCH_SIZE)

    encode_output, last_state = encoder(tmp_x, state)
    print("================== Encoder ==================")
    print("Output encode: ", encode_output.shape)
    print("State_hidden: ", last_state[0].shape)
    print("State_cell: ", last_state[1].shape)

    decoder = Seq2SeqDecode(vocab_size, embedding_size, hidden_unit, n_layers=1)
    decode_output, state = decoder(tmp_x, state=last_state, training=False)
    print("================== Decoder ==================")
    print("Output decode: ", decode_output.shape)
    print("State_hidden: ", state[0].shape)
    print("State_cell: ", state[1].shape)
