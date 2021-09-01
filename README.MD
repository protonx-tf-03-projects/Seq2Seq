# Seq2Seq

## Architecture Image

<b align="center">
<img src="F:\4. PROJECT_TF_03\Seq2Seq\assets\Net.png"/>
</b>

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
                 --batch-size=128 --hidden-units=64 --embedding-size=32 \
                 --epochs=1000  --test-split-size=0.05 --min-sentence=10 \
                 --max-sentence=14 --use-lr-schedule=False --learning_rate=0.005 \
                 --train-mode="not_attention" --debug=True
```

- Use attention

```bash
python train.py  --inp-lang=${path_to_en_text_file} --target-lang=${path_to_vi_text_file} \
                 --batch-size=128 --hidden-units=512 --embedding-size=256 \
                 --epochs=1000 --test-split-size=0.05 --min-sentence=10 \
                 --max-sentence=14 --warmup-steps=100 --train-mode="attention" \
                 --use-lr-schedule=True --debug=True
```

There are some important arguments for the script you should consider when running it:

- `dataset`: The folder of dataset
    - `train.en.txt`: input language
    - `train.vi.txt`: target language

## IV. Predict Process

- Not use attention

```bash
python predict.py --test-path=${link_to_test_data} --inp-lang-path=${link_to_input_data} \
                  --tar-lang-path=${link_to_target_data} --hidden-units=64 \
                  --embedding-size=32 --min-sentence=10 --max-sentence=14 \
                  --train-mode="not_attention"
```

- Use attention

```bash
python predict.py --test-path=${link_to_test_data} --inp-lang-path=${link_to_input_data} \
                  --tar-lang-path=${link_to_target_data} --hidden-units=64 \
                  --embedding-size=32 --min-sentence=10 --max-sentence=14 \
                  --attention-mode="luong" --train-mode="attention"
```

## V. Result and Comparision

Your implementation using Encode-Decode:
<b align="center">
<img src="F:\4. PROJECT_TF_03\Seq2Seq\assets\result_1.png"/>
</b>

Your comments about these results:

- This model is good at sort sequences about 5 to 14 characteristics
