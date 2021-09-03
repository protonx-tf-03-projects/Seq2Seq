# Seq2Seq

<p align="center">
    <img src='https://github.com/protonx-tf-03-projects/Seq2Seq/blob/main/assets/Team.png' width=200 class="center">
</p>

## Architecture Image

<p align="center">
    <img src='https://github.com/protonx-tf-03-projects/Seq2Seq/blob/main/assets/Net.png'>
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
- Click [Here](https://colab.research.google.com/drive/11X9pk2rdBAjXVQugfqxPDezZCuj8_QD9#scrollTo=jqC_yVxZ4qje) to open
  notebook in google colab.

**There are some important arguments for the script you should consider when running it:**

- `dataset`: The folder of dataset
    - `train.en.txt`: input language
    - `train.vi.txt`: target language

## IV. Predict Process

### 1. Not use attention

```bash
python predict.py --test-path=${link_to_test_data} --inp-lang-path=${link_to_input_data} \
                  --tar-lang-path=${link_to_target_data} --hidden-units=128 \
                  --embedding-size=64 --min-sentence=10 --max-sentence=14 \
                  --train-mode="not_attention"
```

### 2. Use attention

```bash
python predict.py --test-path=${link_to_test_data} --inp-lang-path=${link_to_input_data} \
                  --tar-lang-path=${link_to_target_data} --hidden-units=512 \
                  --embedding-size=256 --min-sentence=0 --max-sentence=40 \
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

### 2. Implementation using Encoder - Decoder with Attention Mechanism:

```
-----------------------------------------------------------------
Input    :  <sos> the science behind a climate headline <eos>
Predicted:  khoa học đằng sau một tiêu đề về khí hậu <eos> <eos> <eos> <eos> <eos> <eos> <eos> <eos> <eos> lên <eos> <eos> <eos> <eos> <eos> <eos> <eos> <eos> <eos> <eos>
Target   :  <sos> khoa học đằng sau một tiêu đề về khí hậu <eos>
-----------------------------------------------------------------
Input    :  <sos> they are both two branches of the same field of atmospheric science <eos>
Predicted:  cả hai đều là một nhánh của cùng một lĩnh vực trong ngành khoa học khí quyển <eos> <eos> <eos> <eos> khoa <eos> <eos> lên <eos> <eos> <eos> <eos> <eos>
Target   :  <sos> cả hai đều là một nhánh của cùng một lĩnh vực trong ngành khoa học khí quyển <eos>
-----------------------------------------------------------------
Input    :  <sos> that report was written by 620 scientists from 40 countries <eos>
Predicted:  báo cáo được viết bởi 620 nhà khoa học từ 40 quốc gia <eos> <eos> 40 khoa <eos> <eos> <eos> <eos> từ 40 nước <eos> <eos> <eos> <eos> <eos> <eos>
Target   :  <sos> nghiên cứu được viết bởi 620 nhà khoa học từ 40 quốc gia khác nhau <eos>
-----------------------------------------------------------------
Input    :  <sos> they wrote almost a thousand pages on the topic <eos>
Predicted:  họ viết gần 1000 trang về chủ đề này <eos> đáp <eos> về <eos> <eos> <eos> <eos> <eos> <eos> <eos> <eos> <eos> <eos> <eos> <eos> <eos> <eos> <eos> <eos> <eos>
Target   :  <sos> họ viết gần 1000 trang về chủ đề này <eos>
-----------------------------------------------------------------
Input    :  <sos> and all of those pages were reviewed by another 400 plus scientists and reviewers from 113 countries <eos>
Predicted:  và tất cả những trang đó đều được xem xét lại bởi 400 nhà khoa học và nhà phê bình khác từ từ nhiều quốc gia <eos> <eos> phê từ
Target   :  <sos> và tất cả các trang đều được xem xét bởi 400 khoa học gia và nhà phê bình khác từ 113 quốc gia <eos>
-----------------------------------------------------------------
Input    :  <sos> over 15 000 scientists go to san francisco every year for that <eos>
Predicted:  trong vòng 15 000 nhà khoa học đến san francisco để tham dự hội nghị này <eos> năm nay <eos> năm <eos> <eos> <eos> <eos> <eos> <eos> <eos> <eos> <eos>
Target   :  <sos> mỗi năm hơn 15 000 nhà khoa học đến san francisco để tham dự hội nghị này <eos>
-----------------------------------------------------------------
Epoch 51 -- Loss: 5987.03271484375 -- Bleu_score: 0.5728632331365847
=================================================================
```

**Comments about these results:**

- Model with no attention is good at sort sequences about 5 to 14 characteristics.
- Model with attention needs a lot of data for training.
