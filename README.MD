# Seq2Seq

Implementation of [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/pdf/1409.3215.pdf). This
library is part of our project: Building an AI library with ProtonX.

<p align="center">
    <img src='https://github.com/protonx-tf-03-projects/Seq2Seq/blob/main/assets/Team.png' width=200 class="center">
</p>

In this paper, the authors used `Encoder - Decoder` base model and `English-Franche` dataset, but we
used `Vietnamese-English` dataset and added
two `Attention mechanisms`: [Luong attention](https://arxiv.org/pdf/1508.04025.pdf)
and [Bahdanau attention](https://arxiv.org/pdf/1409.0473.pdf) to choose

**Note:** In attention training part, we used `warm-up learning rate`, it's used in part 5.3
of [Attention Is All You Need](https://arxiv.org/pdf/1706.03762.pdf) paper.

## Architecture Image

<p align="center">
    <img src="https://github.com/protonx-tf-03-projects/Seq2Seq/blob/main/assets/Net.png">
</p>

Authors:

- Github:
    - https://github.com/Xunino
    - https://github.com/khucnam
    - https://github.com/khanhmdds

Advisors:

- Github: https://github.com/bangoc123

## I. Set up environment

- Step 1:

```bash
conda create -n {your_env_name} python==3.7.0
```

- Step 2:

```bash
conda env create -f environment.yml
```

- Step 3:

```bash
conda activate {your_env_name}
``` 

## II. Set up your dataset

- Guide user how to download your data and set the data pipeline
- References: [NLP](https://github.com/protonx-tf-03-projects/Seq2Seq/tree/main/dataset)

## III. Training Process

Training script:

- Not use attention:

```bash
python train.py  --inp-lang=${path_to_en_text_file} --target-lang=${path_to_vi_text_file} \
                 --batch-size=128 --hidden-units=128 --embedding-size=64 \
                 --epochs=1000 --train-mode="not_attention" --test-split-size=0.1 \
                 --min-sentence=10 --max-sentence=14 \
                 --learning-rate=0.005 --bleu=True --debug=True
```

- Use attention

```bash
python train.py  --inp-lang=${path_to_en_text_file} --target-lang=${path_to_vi_text_file} \
                 --batch-size=128 --hidden-units=512 --embedding-size=256 \
                 --epochs=1000 --train-mode="attention" --test-split-size=0.001 \
                 --min-sentence=0 --max-sentence=40 --debug=True \
                 --warmup-steps=150 --use-lr-schedule=True --bleu=True
```

**Note**:

- If you want to retrain model, you can use this param: ```--retrain=True```

- <a href="https://colab.research.google.com/drive/11X9pk2rdBAjXVQugfqxPDezZCuj8_QD9#scrollTo=jqC_yVxZ4qje" target="_blank">
  <img src="https://camo.githubusercontent.com/84f0493939e0c4de4e6dbe113251b4bfb5353e57134ffd9fcab6b8714514d4d1/68747470733a2f2f636f6c61622e72657365617263682e676f6f676c652e636f6d2f6173736574732f636f6c61622d62616467652e737667">

</a>


**There are some important arguments for the script you should consider when running it:**

- `dataset`: The folder of dataset
    - `train.en.txt`: input language
    - `train.vi.txt`: target language

## IV. Predict Process

### 1. Not use attention

```bash
python predict.py --test-path=${link_to_test_data} --hidden-units=128 \
                  --embedding-size=64 --max-sentence=14 \
                  --train-mode="not_attention"
```

### 2. Use attention

```bash
python predict.py --test-path=${link_to_test_data} --hidden-units=512 \
                  --embedding-size=256 --max-sentence=40 \
                  --attention-mode="luong" --train-mode="attention"
```

**Note:**

- If you want to translate a sentence, you can use this param: `--predict-a-sentence=True`

## V. Result and Comparision

### 1. Implementation using Encode - Decode:

```
-----------------------------------------------------------------
Input    :  <sos> they wrote almost a thousand pages on the topic <eos>
Predicted:  <sos> họ viết gần 1000 trang về của tranh của mình <eos> <eos> <eos>
Target   :  <sos> họ viết gần 1000 trang về chủ đề này <eos>
-----------------------------------------------------------------
Input    :  <sos> we blow it up and look at the pieces <eos>
Predicted:  <sos> chúng tôi cho nó nổ và xem xét từng mảnh nhỏ <eos> <eos>
Target   :  <sos> chúng tôi cho nó nổ và xem xét từng mảnh nhỏ <eos>
-----------------------------------------------------------------
Input    :  <sos> this is the euphore smog chamber in spain <eos>
Predicted:  <sos> đây là phòng nghiên cứu khói bụi euphore ở tây ban nha <eos>
Target   :  <sos> đây là phòng nghiên cứu khói bụi euphore ở tây ban nha <eos>
-----------------------------------------------------------------
Input    :  <sos> we also fly all over the world looking for this thing <eos>
Predicted:  <sos> chúng tôi còn bay khắp thế giới để tìm hiểu 50 tiết kiệm
Target   :  <sos> chúng tôi còn bay khắp thế giới để tìm phân tử này <eos>
-----------------------------------------------------------------
Input    :  <sos> this is the tower in the middle of the rainforest from above <eos>
Predicted:  <sos> đây chính là cái tháp giữa rừng sâu nhiều với 100 quốc <eos>
Target   :  <sos> đây chính là cái tháp giữa rừng sâu nhìn từ trên cao <eos>
-----------------------------------------------------------------
Input    :  <sos> christopher decharms a look inside the brain in real time <eos>
Predicted:  <sos> christopher decharms quét não bộ theo thời gian thực <eos> <eos> <eos> <eos>
Target   :  <sos> christopher decharms quét não bộ theo thời gian thực <eos>
=================================================================
Epoch 376 -- Loss: 30.585018157958984 -- Bleu_score: 0.6258203949505547
=================================================================
```

- [Another result](https://github.com/protonx-tf-03-projects/Seq2Seq/blob/main/assets/result_1.png)

### 2. Implementation using Encoder - Decoder with Attention Mechanism:

```
-----------------------------------------------------------------
Input    :  <sos> the science behind a climate headline <eos>
Predicted:  khoa học đằng sau một tiêu đề về khí hậu <eos> <eos> <eos> <eos> <eos> <eos> <eos> <eos> <eos> <eos>
Target   :  <sos> khoa học đằng sau một tiêu đề về khí hậu <eos>
-----------------------------------------------------------------
Input    :  <sos> they are both two branches of the same field of atmospheric science <eos>
Predicted:  cả hai đều là một nhánh của cùng một lĩnh vực trong ngành khoa học khí quyển <eos> <eos> <eos>
Target   :  <sos> cả hai đều là một nhánh của cùng một lĩnh vực trong ngành khoa học khí quyển <eos>
-----------------------------------------------------------------
Input    :  <sos> that report was written by 620 scientists from 40 countries <eos>
Predicted:  nghiên cứu được viết bởi 620 nhà khoa học từ 40 quốc gia khác nhau <eos> <eos> <eos> <eos> <eos>
Target   :  <sos> nghiên cứu được viết bởi 620 nhà khoa học từ 40 quốc gia khác nhau <eos>
-----------------------------------------------------------------
Input    :  <sos> they wrote almost a thousand pages on the topic <eos>
Predicted:  họ viết gần 1000 trang về chủ đề này <eos> kê trên cùng <eos> <eos> <eos> <eos> <eos> <eos> <eos>
Target   :  <sos> họ viết gần 1000 trang về chủ đề này <eos>
-----------------------------------------------------------------
Input    :  <sos> over 15 000 scientists go to san francisco every year for that <eos>
Predicted:  mỗi năm hơn 15 000 nhà khoa học đến san francisco để tham dự hội nghị này <eos> vũ <eos>
Target   :  <sos> mỗi năm hơn 15 000 nhà khoa học đến san francisco để tham dự hội nghị này <eos>
-----------------------------------------------------------------
Input    :  <sos> it apos s a huge amount of stuff it apos s equal to the weight of methane <eos>
Predicted:  đó là một lượng khí thải khổng lồ bằng tổng trọng lượng của mêtan <eos> <eos> <eos> <eos> <eos> <eos>
Target   :  <sos> đó là một lượng khí thải khổng lồ bằng tổng trọng lượng của mêtan <eos>
-----------------------------------------------------------------
Epoch 93 -- Loss: 295.7635192871094 -- Bleu_score: 0.956022394501728
=================================================================
```

- [Another result](https://github.com/protonx-tf-03-projects/Seq2Seq/blob/main/assets/result_2.png)

**Comments about these results:**

- Model with no attention is good at sort sequences about 5 to 14 characteristics.
- Model with attention needs a lot of data for training.
