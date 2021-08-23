# Define constant variables
import collections
from typing import List

import numpy as np


def remove_oov(sentence):
    return [i for i in sentence.split(" ") if i not in ["<eos>", "<sos>"]]


# n-grams
# Tìm đoạn đúng dài nhất trong câu dự đoán trùng với trong câu đích
def bleu_score(predicted_sentence, target_sentences, n_grams=3):
    predicted_sentence = remove_oov(predicted_sentence)
    target_sentences = remove_oov(target_sentences)
    pred_length = len(predicted_sentence)
    target_length = len(target_sentences)
    score = np.exp(np.minimum(0, 1 - target_length / pred_length))
    for k in range(1, n_grams + 1):
        label_subs = collections.defaultdict(int)
        num_matches = 0
        for i in range(target_length - k + 1):
            label_subs[" ".join(target_sentences[i:i + k])] += 1
        for i in range(pred_length - k + 1):
            if label_subs[" ".join(predicted_sentence[i:i + k])] > 0:
                label_subs[" ".join(predicted_sentence[i:i + k])] -= 1
                num_matches += 1
        score *= np.power(num_matches / (pred_length - k + 1), np.power(0.5, k))
    return score


if __name__ == '__main__':
    pred = "<sos> và đó là rằng chúng ta có thể thấy những người ở đây là gì <eos> <eos> <eos> <eos>"
    target = "<sos> và đó là con đường chúng ta tiến tới trong kinh doanh <eos>"

    print(bleu_score(pred, target, n_grams=3))
