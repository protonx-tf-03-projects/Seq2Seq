import string
import re
import os
import numpy as np


class BuildLanguage:
    def __init__(self, lines):
        self.lines = lines
        self.word2id = {}
        self.id2word = {}
        self.vocab = set()
        self.max_len = 0
        self.min_len = 0
        self.vocab_size = 0
        self.init_language_params()

    def init_language_params(self):
        for line in self.lines:
            self.vocab.update(line.split(" "))

        self.word2id['<pad>'] = 0

        for id, word in enumerate(self.vocab):
            self.word2id[word] = id + 1

        for word, id in self.word2id.items():
            self.id2word[id] = word

        self.max_len = max([len(line.split(" ")) for line in self.lines])
        self.min_len = min([len(line.split(" ")) for line in self.lines])

        self.vocab_size = len(self.vocab) + 1

    def sentence_to_vector(self, sent):
        return np.array([self.word2id[word] for word in sent.split(" ")])

    def vector_to_sentence(self, vector):
        sentence = []
        for id in vector:
            if id == 0:
                break
            sentence.append(self.id2word[id])
        return " ".join(sentence)


class DatasetLoader:
    """
    :input:
        Khởi tạo dữ liệu cho quá trình huấn luyện, bao gồm 2 tập.
            1. train.tv.txt : Dữ liệu ngôn ngữ gốc (Tiếng Việt)
            2. train.ta.txt : Dữ liệu ngôn ngữ chuyển đổi (Tiếng Anh)

    :doing:
        1. Khởi tạo liệu
        2. Xóa dấu câu và số
        3. Thêm phần tử nhận diện lúc bắt đầu và kết thúc dịch (VD: <start>, <stop>, ...)
        4. Xử lý độ dài câu: min_length <= length <= max_length

    :return:
        Dữ liệu sau khi tiền xử lý: list
    """

    def __init__(self,
                 language_1,
                 language_2,
                 min_length=10,
                 max_length=14):
        """
            Khởi tạo

        :param language_1: ${}/train.{original}.txt ---- Đường dẫn tới ngôn ngữ gốc
        :param language_2: ${}/train.{target}.txt ---- Đường dẫn tối ngôn ngữ chuyển đổi

        :param min_length: Giới hạn nhỏ nhất chiều dài 1 câu
        :param max_length: Giới hạn lớn nhất chiều dài 1 câu
        """

        self.language_1 = language_1
        self.language_2 = language_2

        self.min_length = min_length
        self.max_length = max_length

        self.punctuation_digits = list(string.punctuation + string.digits)

    def load_dataset(self):
        """
        :doing:
            Load data from direction

        :return: Trả về dữ liệu dạng list
        """
        current_dir = os.getcwd() + "/"
        raw_origin_language = open(current_dir + self.language_1, encoding="UTF-8").read().strip().split("\n")
        raw_target_language = open(current_dir + self.language_2, encoding="UTF-8").read().strip().split("\n")
        assert len(raw_target_language) == len(raw_origin_language)

        return raw_origin_language, raw_target_language

    def build_dataset(self, debug=False):
        """
        :doing:
            1. Khởi tạo liệu
            2. Xóa dấu câu và số
            3. Thêm phần tử nhận diện lúc bắt đầu và kết thúc dịch (VD: <start>, <stop>, ...)
            4. Xử lý độ dài câu: min_length <= length <= max_length
        :return:
        """
        # Khởi tạo dữ liệu
        raw_origin_language, raw_target_language = self.load_dataset()

        # Xóa dấu câu và số
        # Thêm phần tử nhận diện lúc bắt đầu và kết thúc dịch (VD: <start>, <stop>, ...)
        # Xử lý độ dài câu: min_length <= length <= max_length
        inp_lang, tar_lang = self.preprocessing_sentence(raw_origin_language, raw_target_language)

        inp_vector = [inp_lang.sentence_to_vector(line) for line in inp_lang.lines]
        tar_vector = [tar_lang.sentence_to_vector(line) for line in tar_lang.lines]

        if debug:
            for vector, sentence in zip(inp_vector[:5], inp_lang.lines[:5]):
                print("Vector: {}\nSentence: {}\n\n".format(vector, sentence))

        return inp_vector, tar_vector, inp_lang, tar_lang

    def remove_punctuation_digits(self, sen):
        """
        :input: sen: str

        :doing:
            1. Xóa dấu câu và số
            2. Thêm phần tử nhận diện lúc bắt đầu và kết thúc dịch (VD: <start>, <stop>, ...)

        :return:
            Dữ liệu không chứa dấu câu và số
        """
        sen = sen.lower()
        sen = sen.strip()
        sen = re.sub("'", "", sen)
        sen = re.sub("\s+", " ", sen)
        # sen = " ".join([s for s in sen.split(" ") if s not in self.punctuation_digits])
        return "<sos> " + sen + " <eos>"

    def preprocessing_sentence(self, raw_origin_language, raw_target_language):
        """
        :input:
            language_1: Ngôn ngữ gốc: (list)
            language_2: Ngôn ngữ mục tiêu: (list)

        :doing:
            1. Xử lý độ dài câu: min_length <= length <= max_length

        :return:
        """
        sentences_1 = []
        sentences_2 = []
        for sen_1, sen_2 in zip(raw_origin_language, raw_target_language):
            sen_1 = self.remove_punctuation_digits(sen_1)
            sen_2 = self.remove_punctuation_digits(sen_2)
            if self.min_length <= len(sen_1.split(" ")) <= self.max_length \
                    and self.min_length <= len(sen_2.split(" ")) <= self.max_length:
                sentences_1.append(sen_1)
                sentences_2.append(sen_2)

        inp_lang = BuildLanguage(sentences_1)
        tar_lang = BuildLanguage(sentences_2)

        return inp_lang, tar_lang


if __name__ == '__main__':
    data = DatasetLoader("dataset/train.vi.txt", "dataset/train.en.txt")
    processed_original_language, processed_target_language, _, _ = data.build_dataset(True)
