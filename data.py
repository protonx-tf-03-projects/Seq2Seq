import json
import string
import re
import os
from tensorflow.keras.preprocessing.text import Tokenizer


def remove_punctuation(sen):
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
    sen = " ".join([s for s in sen.split() if s not in list(string.punctuation)])
    return "<sos> " + sen + " <eos>"


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

        self.save_dict = os.getcwd() + "/saved_models/{}_vocab.json"

    def save_tokenizer(self, object, name_vocab):
        f = open(self.save_dict.format(name_vocab), "w", encoding="utf-8")
        json.dumps(f.write(str(object.word_index).replace("'", "\"")))
        f.close()

    def load_tokenizer(self, name_vocab):
        f = open(self.save_dict.format(name_vocab), "r", encoding="utf-8")
        return json.load(f)

    def load_dataset(self):
        """
        :doing:
            Load data from direction

        :return: Trả về dữ liệu dạng list
        """
        current_dir = os.getcwd() + "/"
        raw_origin_language = open(current_dir + self.language_1, encoding="UTF-8").read().strip().split("\n")
        raw_target_language = open(current_dir + self.language_2, encoding="UTF-8").read().strip().split("\n")

        return raw_origin_language, raw_target_language

    def build_dataset(self):
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

        # Build Tokenizer
        tokenize_inp = Tokenizer(filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')
        tokenize_tar = Tokenizer(filters='!"#$%&()*+,-./:;=?@[\\]^_`{|}~\t\n')

        # Fit text
        tokenize_inp.fit_on_texts(inp_lang)
        tokenize_tar.fit_on_texts(tar_lang)

        # save tokenizer
        self.save_tokenizer(tokenize_inp, name_vocab="input")
        self.save_tokenizer(tokenize_tar, name_vocab="target")

        # Get tensor
        inp_vector = tokenize_inp.texts_to_sequences(inp_lang)
        tar_vector = tokenize_tar.texts_to_sequences(tar_lang)

        return inp_vector, tar_vector, tokenize_inp, tokenize_tar

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
            sen_1 = remove_punctuation(sen_1)
            sen_2 = remove_punctuation(sen_2)
            if self.min_length <= len(sen_1.split(" ")) <= self.max_length \
                    and self.min_length <= len(sen_2.split()) <= self.max_length:
                sentences_1.append(sen_1)
                sentences_2.append(sen_2)

        return sentences_1, sentences_2


if __name__ == '__main__':
    save_dict = os.getcwd() + "/saved_models/{}_vocab.json"
    f = open(save_dict.format("input"), "r", encoding="utf-8")
    a = json.load(f)
    # print(a.values())
