# Seq2Seq

<p align="center">
    <img src='https://storage.googleapis.com/protonx-cloud-storage/transformer/protonx-transf.png' width=200 class="center">
</p>

Description about your project. Why do you choose to build this?  <--- **FIXME**

Slide about your project (if it's available) <--- **FIXME**

Architecture Image <--- **FIXME**


Authors:
- Github: members github name <--- **FIXME**
- Email: members emails <--- **FIXME**

Advisors:
- Github: advisor github name <--- **FIXME**
- Email: advisor emails <--- **FIXME**

## I.  Set up environment
- Step 1: 

```python
conda create -n {your_env_name} python==3.7.0
```

- Step 2:
```python
conda env create -f environment.yml
```

- Step 3:
```python
conda activate {your_env_name}
``` 

## II.  Set up your dataset

- Guide user how to download your data and set the data pipeline
- References: [NLP](https://github.com/protonx-tf-03-projects/Seq2Seq/tree/main/dataset)

## III. Training Process


**FIXME**

Training script:


```python

python train.py --epochs=${epochs} --inp-lang=${path_to_en_text_file} --target-lang=${path_to_vi_text_file} \
                --batch-size=128  --hidden-units=1024 --embedding-size=512 --min-sentence=10 --max-sentence=14 \
                --epochs=1000 --train-mode="attention" --test-split-size=0.001 --learning-rate=0.005 --debug=True
                 

```
**FIXME**

Example:

```python

!python train.py --epochs=${epochs} --inp-lang=${path_to_en_text_file} --target-lang=${path_to_vi_text_file} \
                 --batch-size=128  --hidden-units=1024 --embedding-size=512 --min-sentence=10 --max-sentence=14 \
                 --epochs=1000 --train-mode="attention" --test-split-size=0.001 --learning-rate=0.005 --debug=True
``` 
**FIXME**

There are some important arguments for the script you should consider when running it:

- `dataset`: The folder of dataset
  - `train.en.txt`: input language
  - `train.vi.txt`: target language

## IV. Predict Process

```bash
python predict.py --test-data ${link_to_test_data}
```

## V. Result and Comparision

**FIXME**

Your implementation using Encode-Decode:
<b align="center">
<img src="F:\4. PROJECT_TF_03\Seq2Seq\assets\image.png"/>
</b>

**FIXME**

Other architecture

```
Epoch 6/10
391/391 [==============================] - 115s 292ms/step - loss: 0.1999 - acc: 0.9277 - val_loss: 0.4719 - val_acc: 0.8130
Epoch 7/10
391/391 [==============================] - 114s 291ms/step - loss: 0.1526 - acc: 0.9494 - val_loss: 0.5224 - val_acc: 0.8318
Epoch 8/10
391/391 [==============================] - 115s 293ms/step - loss: 0.1441 - acc: 0.9513 - val_loss: 0.5811 - val_acc: 0.7875
```

Your comments about these results <--- **FIXME**


## VI. Running Test

When you want to modify the model, you need to run the test to make sure your change does not affect the whole system.

In the `./folder-name` **(FIXME)** folder please run:

```bash
pytest
```


