import string
import re
import tensorflow as tf


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
                 min_length=20,
                 max_length=30):
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
        raw_origin_language = open(self.language_1, encoding="UTF-8").read().strip().split("\n")
        raw_target_language = open(self.language_2, encoding="UTF-8").read().strip().split("\n")
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
        processed_original, processed_target = self.preprocessing_sentence(raw_origin_language,
                                                                           raw_target_language)
        if debug:
            for sen_1, sen_2 in zip(processed_original[:5], processed_target[:5]):
                print("{}\n{}\n\n".format(sen_1, sen_2))

        return processed_original, processed_target

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
        sen = " ".join([s for s in sen.split(" ") if s not in self.punctuation_digits])
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
        return sentences_1, sentences_2


if __name__ == '__main__':
    data = Dataset("dataset/train.vi.txt", "dataset/train.en.txt")
    processed_original_language, processed_target_language = data.build_dataset(True)
    # print(len(processed_original_language), len(processed_target_language))
