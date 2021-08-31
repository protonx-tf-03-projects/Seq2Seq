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
        return [i for i in sentence.split(" ") if i not in ["<eos>", "<sos>"]]

    def __call__(self, predicted_sentence, target_sentences, n_grams=3):
        predicted_sentence = self.remove_oov(predicted_sentence)
        target_sentences = self.remove_oov(target_sentences)
        pred_length = len(predicted_sentence)
        target_length = len(target_sentences)

        if pred_length < n_grams:
            return 0
        else:
            score = np.exp(np.minimum(0, 1 - target_length / pred_length))
            for k in range(1, n_grams + 1):
                label_subs = collections.defaultdict(int)
                for i in range(target_length - k + 1):
                    label_subs[" ".join(target_sentences[i:i + k])] += 1

                num_matches = 0
                for i in range(pred_length - k + 1):
                    if label_subs[" ".join(predicted_sentence[i:i + k])] > 0:
                        label_subs[" ".join(predicted_sentence[i:i + k])] -= 1
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
