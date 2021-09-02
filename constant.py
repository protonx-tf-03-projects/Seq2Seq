import numpy as np
import collections
import tensorflow as tf


class Bleu_score:
    """
        We can evaluate a predicted sequence by comparing it with the label sequence.
        BLEU (Bilingual Evaluation Understudy) "https://aclanthology.org/P02-1040.pdf",
        though originally proposed for evaluating machine translation results,
        has been extensively used in measuring the quality of output sequences for different applications.
        In principle, for any n-grams in the predicted sequence, BLEU evaluates whether this n-grams appears
        in the label sequence.
    """

    def __init__(self):
        super().__init__()

    def remove_oov(self, sentence):
        return [i for i in sentence.split(" ") if i not in ["<sos>", "<eos>"]]

    def __call__(self, pred, target, n_grams=3):
        pred = self.remove_oov(pred)
        target = self.remove_oov(target)
        pred_length = len(pred)
        target_length = len(target)

        if pred_length < n_grams:
            return 0
        else:
            score = np.exp(np.minimum(0, 1 - target_length / pred_length))
            for k in range(1, n_grams + 1):
                label_subs = collections.defaultdict(int)
                for i in range(target_length - k + 1):
                    label_subs[" ".join(target[i:i + k])] += 1

                num_matches = 0
                for i in range(pred_length - k + 1):
                    if label_subs[" ".join(pred[i:i + k])] > 0:
                        label_subs[" ".join(pred[i:i + k])] -= 1
                        num_matches += 1
                score *= np.power(num_matches / (pred_length - k + 1), np.power(0.5, k))
            return score


class MaskedSoftmaxCELoss(tf.keras.losses.Loss):
    """
        The softmax cross-entropy loss with masks.

        Tính giá trị mất mát tập chung các vị trí xuất hiện (Y_hat) từ giống với giá trị gốc (Y_true):

        Ví dụ cơ bản:
            Đầu vào:
                pred_matrix = [1, 25, 1445, 105, 5, 4, 8, 2]
                true_matrix = [0, 20, 1456, 145, 2, 0, 0, 0]

            Chỗ nào có giá trị khác 0 tại true_matrix gán bằng giá trị 1:
            mask_matrix = [0, 1, 1, 1, 1, 0, 0, 0]

            loss = true_matrix - pred_matrix = [1, 0, 11, 40, 0, 4, 8, 2]
            ==> loss = loss * mask_matrix = [0, 5, 11, 40, 3, 0, 0, 0]
    """

    def __init__(self):
        super().__init__()

    def call(self, label, pred):
        """
        :param label: shape (batch_size, max_length, vocab_size)
        :param pred: shape (batch_size, max_length)

        :return: weighted_loss: shape (batch_size, max_length)
        """

        weights_mask = 1 - np.equal(label, 0)
        unweighted_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(label, pred)
        weighted_loss = tf.reduce_mean(unweighted_loss * weights_mask)
        return weighted_loss


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
        Following with learning rate schedule in paper: https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, hidden_units, warmup_steps=250):
        super(CustomSchedule, self).__init__()

        self.hidden_units = hidden_units
        self.hidden_units = tf.cast(self.hidden_units, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.hidden_units) * tf.math.minimum(arg1, arg2)


def evaluation(model,
               test_ds,
               val_function,
               inp_builder,
               tar_builder,
               test_split_size,
               debug=False):
    """
    :param test_ds: (inp_vocab, tar_vocab)
    :param (inp_lang, tar_lang)
    :return:
    """
    # Preprocessing testing data
    score = 0.0
    count = 0
    test_ds_len = int(len(test_ds) * test_split_size)
    for test_, test_y in test_ds.shuffle(buffer_size=1, seed=1).take(test_ds_len):
        test_x = tf.expand_dims(test_, axis=0)
        _, last_state = model.encoder(test_x)

        input_decode = tf.reshape(tf.constant([tar_builder.word_index['<sos>']]), shape=(-1, 1))
        sentence = []
        for _ in range(len(test_y)):
            output, last_state = model.decoder(input_decode, last_state, training=False)
            output = tf.argmax(output, axis=2).numpy()
            input_decode = output
            sentence.append(output[0][0])

        input_sequence = inp_builder.sequences_to_texts([test_.numpy()])[0]
        pred_sequence = tar_builder.sequences_to_texts([sentence])[0]
        target_sequence = tar_builder.sequences_to_texts([test_y.numpy()])[0]
        score += val_function(pred_sequence,
                              target_sequence)
        if debug and count <= 5:
            print("-----------------------------------------------------------------")
            print("Input    : ", input_sequence)
            print("Predicted: ", pred_sequence)
            print("Target   : ", target_sequence)
            count += 1
    return score / test_ds_len


def evaluation_with_attention(model,
                              test_ds,
                              val_function,
                              inp_builder,
                              tar_builder,
                              test_split_size,
                              debug=False):
    """
    :param test_ds: (inp_vocab, tar_vocab)
    :param (inp_lang, tar_lang)
    :return:
    """
    # Preprocessing testing data
    score = 0.0
    count = 0
    test_ds_len = int(len(test_ds) * test_split_size)
    for test_, test_y in test_ds.shuffle(buffer_size=1, seed=1).take(test_ds_len):
        test_x = tf.expand_dims(test_, axis=0)
        encode_outs, last_state = model.encoder(test_x)
        input_decode = tf.constant([tar_builder.word_index['<sos>']])
        sentence = []
        for _ in range(len(test_y)):
            output, last_state = model.decoder(input_decode, encode_outs, last_state, training=False)
            pred_id = tf.argmax(output, axis=1).numpy()
            input_decode = pred_id
            sentence.append(pred_id[0])

        input_sequence = inp_builder.sequences_to_texts([test_.numpy()])[0]
        pred_sequence = tar_builder.sequences_to_texts([sentence])[0]
        target_sequence = tar_builder.sequences_to_texts([test_y.numpy()])[0]

        score += val_function(pred_sequence,
                              target_sequence)
        if debug and count <= 5:
            print("-----------------------------------------------------------------")
            print("Input    : ", input_sequence)
            print("Predicted: ", pred_sequence)
            print("Target   : ", target_sequence)
            count += 1
    return score / test_ds_len


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    temp_learning_rate_schedule = CustomSchedule(64, 100)

    plt.plot(temp_learning_rate_schedule(tf.range(1000, dtype=tf.float32)))
    plt.ylabel("Learning Rate")
    plt.xlabel("Train Step")
    plt.show()
