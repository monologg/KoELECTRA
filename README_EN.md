[한국어](./README.md) | [English](./README_EN.md)

# KoELECTRA

<p float="left" align="center">
    <img width="900" src="https://user-images.githubusercontent.com/28896432/80024445-0f444e00-851a-11ea-9137-9da2abfd553d.png" />  
</p>

[ELECTRA](https://openreview.net/pdf?id=r1xMH1BtvB) uses `Replaced Token Detection`, in other words, it learns by looking at the token from the generator and determining whether it is a "real" token or a "fake" token in the discriminator. This methods allows to train all input tokens, which shows competitive result compare to other pretrained language models (BERT etc.)

KoELECTRA is trained with **34GB Korean text**, and I'm releasing `KoELECTRA-Base` and `KoELECTRA-Small`.

Also KoELECTRA **uses Wordpiece** and **model is uploaded on s3**, so
just install the `Transformers` library and it will be ready to use regardless of the OS you use.

## Updates

**April 27, 2020**

- Add two additional subtasks (`KorSTS`, `QuestionPair`), and the results were updated for the existing 5 subtasks.

**June 3, 2020**

- `KoELECTRA-v2` is released for both base and small model, which is trained with new vocabulary that is used in [EnlipleAI PLM](https://github.com/enlipleai/kor_pratrain_LM). Both Base and Small models showed improved performance in `KorQuaD`.

**October 9, 2020**

- `KoELECTRA-v3` was produced by additionally using `Everyone's Corpus`. Vocab was also newly produced using `Mecab` and `Wordpiece`.
- In consideration of the official support of `ElectraForSequenceClassification` of `Huggingface Transformers`, the existing subtask results have been updated. Also the result of [Korean-Hate-Speech](https://github.com/kocohub/korean-hate-speech) is added.

```python
from transformers import ElectraModel, ElectraTokenizer

model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
```

## Download Link

| Model                |                                                                     Discriminator |                                                                 Generator |                                                                                       Tensorflow-v1 |
| -------------------- | --------------------------------------------------------------------------------: | ------------------------------------------------------------------------: | --------------------------------------------------------------------------------------------------: |
| `KoELECTRA-Base-v1`  |     [Discriminator](https://huggingface.co/monologg/koelectra-base-discriminator) |     [Generator](https://huggingface.co/monologg/koelectra-base-generator) | [Tensorflow-v1](https://drive.google.com/file/d/1AnyoxEG0nI7NZM7luy7yCL_3_8h00Oaw/view?usp=sharing) |
| `KoELECTRA-Small-v1` |    [Discriminator](https://huggingface.co/monologg/koelectra-small-discriminator) |    [Generator](https://huggingface.co/monologg/koelectra-small-generator) | [Tensorflow-v1](https://drive.google.com/file/d/1P9ry0g9NbqUBd7X8WbHIcB627EodpDb6/view?usp=sharing) |
| `KoELECTRA-Base-v2`  |  [Discriminator](https://huggingface.co/monologg/koelectra-base-v2-discriminator) |  [Generator](https://huggingface.co/monologg/koelectra-base-v2-generator) | [Tensorflow-v1](https://drive.google.com/file/d/1i028LR4BIa0c5z6o03gzrb67KqR_74FV/view?usp=sharing) |
| `KoELECTRA-Small-v2` | [Discriminator](https://huggingface.co/monologg/koelectra-small-v2-discriminator) | [Generator](https://huggingface.co/monologg/koelectra-small-v2-generator) | [Tensorflow-v1](https://drive.google.com/file/d/1y_SKg9XT5dsDXXElo8DmZuUk1sRXR8p5/view?usp=sharing) |
| `KoELECTRA-Base-v3`  |  [Discriminator](https://huggingface.co/monologg/koelectra-base-v3-discriminator) |  [Generator](https://huggingface.co/monologg/koelectra-base-v3-generator) | [Tensorflow-v1](https://drive.google.com/file/d/1L8TChlO0_bNJCHNAV3m7al-iaOt1EWkY/view?usp=sharing) |
| `KoELECTRA-Small-v3` | [Discriminator](https://huggingface.co/monologg/koelectra-small-v3-discriminator) | [Generator](https://huggingface.co/monologg/koelectra-small-v3-generator) | [Tensorflow-v1](https://drive.google.com/file/d/1qFVIaCdGXQMlS0MEQlgOWxjk-cVG75Qu/view?usp=sharing) |

## About KoELECTRA

|                   |               | Layers | Embedding Size | Hidden Size | # heads |
| ----------------- | ------------: | -----: | -------------: | ----------: | ------: |
| `KoELECTRA-Base`  | Discriminator |     12 |            768 |         768 |      12 |
|                   |     Generator |     12 |            768 |         256 |       4 |
| `KoELECTRA-Small` | Discriminator |     12 |            128 |         256 |       4 |
|                   |     Generator |     12 |            128 |         256 |       4 |

### Vocabulary

- The main purpose of this project was to **make the model immediately available with the Transformers library**, therefore, instead of using the Sentencepiece and Mecab, the `Wordpiece` used in the original paper and code was used.
- For more detail, see [[Wordpiece Vocabulary]](./docs/wordpiece_vocab_EN.md)

|     | Vocab Len | do_lower_case |
| --- | --------: | ------------: |
| v1  |     32200 |         False |
| v2  |     32200 |         False |
| v3  |     35000 |         False |

### Data

- For `v1` and `v2`, **14G Corpus** (2.6B tokens) was used. (News, Wiki, Namu Wiki)
- For `v3`, **20G Corpus** from `Everyone's Corpus` was additionally used. (Newspaper, written, spoken, messenger, web)
- For more detail, see [[Preprocessing]](./docs/preprocessing_EN.md)

### Pretraining Details

| Model        | Batch Size | Train Steps |   LR | Max Seq Len | Generator Size | Train Time |
| :----------- | ---------: | ----------: | ---: | ----------: | -------------: | ---------: |
| `Base v1,2`  |        256 |        700K | 2e-4 |         512 |           0.33 |         7d |
| `Base v3`    |        256 |        1.5M | 2e-4 |         512 |           0.33 |        14d |
| `Small v1,2` |        512 |        300K | 5e-4 |         512 |            1.0 |         3d |
| `Small v3`   |        512 |        800K | 5e-4 |         512 |            1.0 |         7d |

- In case of `KoELECTRA-Small` model, the same options as `ELECTRA-Small++` in the original paper were used.

  - This is the same setting as the small model distributed by the official ELECTRA code.
  - Also, unlike `KoELECTRA-Base`, the model size of Generator and Disciminator is same.

- Except for `Batch size` and `Train steps`, other hyperparameters are same as that of original paper.

  - I tried changing other hyperparameters and running them, but setting them as same as the original paper performed best.

- **TPU v3-8** was used for pretraining. More detail about using TPU on GCP, see [[Using TPU for Pretraining]](./docs/tpu_training_EN.md)

## KoELECTRA on 🤗 Transformers 🤗

- `ElectraModel` is officially supported from `Transformers v2.8.0`.

- `ElectraModel` is similar to `BertModel` except that it does not return `pooled_output`.

- ELECTRA uses `discriminator` for finetuning.

### 1. Pytorch Model & Tokenizer

```python
from transformers import ElectraModel, ElectraTokenizer

model = ElectraModel.from_pretrained("monologg/koelectra-base-discriminator")  # KoELECTRA-Base
model = ElectraModel.from_pretrained("monologg/koelectra-small-discriminator")  # KoELECTRA-Small
model = ElectraModel.from_pretrained("monologg/koelectra-base-v2-discriminator")  # KoELECTRA-Base-v2
model = ElectraModel.from_pretrained("monologg/koelectra-small-v2-discriminator")  # KoELECTRA-Small-v2
model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")  # KoELECTRA-Base-v3
model = ElectraModel.from_pretrained("monologg/koelectra-small-v3-discriminator")  # KoELECTRA-Small-v3
```

### 2. Tensorflow v2 Model

```python
from transformers import TFElectraModel

model = TFElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator", from_pt=True)
```

### 3. Tokenizer Example

```python
>>> from transformers import ElectraTokenizer
>>> tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
>>> tokenizer.tokenize("[CLS] 한국어 ELECTRA를 공유합니다. [SEP]")
['[CLS]', '한국어', 'EL', '##EC', '##TRA', '##를', '공유', '##합니다', '.', '[SEP]']
>>> tokenizer.convert_tokens_to_ids(['[CLS]', '한국어', 'EL', '##EC', '##TRA', '##를', '공유', '##합니다', '.', '[SEP]'])
[2, 11229, 29173, 13352, 25541, 4110, 7824, 17788, 18, 3]
```

## Result on Subtask

**This is the result of running with the config as it is, and if hyperparameter tuning is additionally performed, better performance may come out.**

For code and more detail, see [[Finetuning]](./finetune/README_EN.md)

### Base Model

|                       | **NSMC**<br/>(acc) | **Naver NER**<br/>(F1) | **PAWS**<br/>(acc) | **KorNLI**<br/>(acc) | **KorSTS**<br/>(spearman) | **Question Pair**<br/>(acc) | **KorQuaD (Dev)**<br/>(EM/F1) | **Korean-Hate-Speech (Dev)**<br/>(F1) |
| :-------------------- | :----------------: | :--------------------: | :----------------: | :------------------: | :-----------------------: | :-------------------------: | :---------------------------: | :-----------------------------------: |
| KoBERT                |       89.59        |         87.92          |       81.25        |        79.62         |           81.59           |            94.85            |         51.75 / 79.15         |                 66.21                 |
| XLM-Roberta-Base      |       89.03        |         86.65          |       82.80        |        80.23         |           78.45           |            93.80            |         64.70 / 88.94         |                 64.06                 |
| HanBERT               |       90.06        |         87.70          |       82.95        |        80.32         |           82.73           |            94.72            |         78.74 / 92.02         |               **68.32**               |
| KoELECTRA-Base        |       90.33        |         87.18          |       81.70        |        80.64         |           82.00           |            93.54            |         60.86 / 89.28         |                 66.09                 |
| KoELECTRA-Base-v2     |       89.56        |         87.16          |       80.70        |        80.72         |           82.30           |            94.85            |         84.01 / 92.40         |                 67.45                 |
| **KoELECTRA-Base-v3** |     **90.63**      |       **88.11**        |     **84.45**      |      **82.24**       |         **85.53**         |          **95.25**          |       **84.83 / 93.45**       |                 67.61                 |

### Small Model

|                        | **NSMC**<br/>(acc) | **Naver NER**<br/>(F1) | **PAWS**<br/>(acc) | **KorNLI**<br/>(acc) | **KorSTS**<br/>(spearman) | **Question Pair**<br/>(acc) | **KorQuaD (Dev)**<br/>(EM/F1) | **Korean-Hate-Speech (Dev)**<br/>(F1) |
| :--------------------- | :----------------: | :--------------------: | :----------------: | :------------------: | :-----------------------: | :-------------------------: | :---------------------------: | :-----------------------------------: |
| DistilKoBERT           |       88.60        |         84.65          |       60.50        |        72.00         |           72.59           |            92.48            |         54.40 / 77.97         |                 60.72                 |
| KoELECTRA-Small        |       88.83        |         84.38          |       73.10        |        76.45         |           76.56           |            93.01            |         58.04 / 86.76         |                 63.03                 |
| KoELECTRA-Small-v2     |       88.83        |         85.00          |       72.35        |        78.14         |           77.84           |            93.27            |         81.43 / 90.46         |                 60.14                 |
| **KoELECTRA-Small-v3** |     **89.36**      |       **85.40**        |     **77.45**      |      **78.60**       |         **80.79**         |          **94.85**          |       **82.11 / 91.13**       |               **63.07**               |

## Acknowledgement

KoELECTRA was created with Cloud TPU support from the **Tensorflow Research Cloud (TFRC)** program. Also, `KoELECTRA-v3` was produced with the help of **Everyone's Corpus**.

## Citation

If you use this code for research, please cite:

```
@misc{park2020koelectra,
  author = {Park, Jangwon},
  title = {KoELECTRA: Pretrained ELECTRA Model for Korean},
  year = {2020},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/monologg/KoELECTRA}}
}
```

## Reference

- [ELECTRA](https://github.com/google-research/electra)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [Tensorflow Research Cloud](https://www.tensorflow.org/tfrc?hl=ko)
- [Chinese ELECTRA](https://github.com/ymcui/Chinese-ELECTRA/blob/master/README_EN.md)
- [Enliple AI Korean PLM](https://github.com/enlipleai/kor_pratrain_LM)
- [Everyone's Corpus](https://corpus.korean.go.kr/)
