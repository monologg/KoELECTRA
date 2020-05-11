[한국어](./README.md) | [English](./README_EN.md)

# KoELECTRA

<p float="left" align="center">
    <img width="900" src="https://user-images.githubusercontent.com/28896432/80024445-0f444e00-851a-11ea-9137-9da2abfd553d.png" />  
</p>

[ELECTRA](https://openreview.net/pdf?id=r1xMH1BtvB) uses `Replaced Token Detection`, in other words, it learns by looking at the token from the generator and determining whether it is a "real" token or a "fake" token in the discriminator. This methods allows to train all input tokens, which shows competitive result compare to other pretrained language models (BERT etc.)

KoELECTRA is trained with **14GB Korean text** (96M sentences, 2.6B tokens), and I'm releasing `KoELECTRA-Base` and `KoELECTRA-Small`.

Also KoELECTRA **uses Wordpiece** and **model is uploaded on s3**, so
just install the `Transformers` library and it will be ready to use regardless of the OS you use.

## Updates

**April 27, 2020** - Add two additional subtasks (`KorSTS`, `QuestionPair`), and the results were updated for the existing 5 subtasks.

## About KoELECTRA

|                   |               | Layers | Embedding Size | Hidden Size | # heads | Size |
| ----------------- | ------------: | -----: | -------------: | ----------: | ------: | ---: |
| `KoELECTRA-Base`  | Discriminator |     12 |            768 |         768 |      12 | 423M |
|                   |     Generator |     12 |            768 |         256 |       4 | 134M |
| `KoELECTRA-Small` | Discriminator |     12 |            128 |         256 |       4 |  53M |
|                   |     Generator |     12 |            128 |         256 |       4 |  53M |

### Vocabulary

The main purpose of this project was to **make the model immediately available with the Transformers library**, therefore, instead of using the Sentencepiece and Mecab, the `Wordpiece` used in the original paper and code was used.

- Size of the Vocabulary is `32200`, which includes 200 `[unused]` tokens
- Cased (`do_lower_case=False`)

For more detail, see [[Wordpiece Vocabulary]](./docs/wordpiece_vocab_EN.md)

### Pretraining Details

- For data, I used **14G Korean Corpus** (2.6B tokens), which has been pre-processed. (For more detail, see [[Preprocessing]](./docs/preprocessing_EN.md))

  |       Model       | Batch Size | Train Steps | Learning Rate | Max Seq Len | Generator Size |
  | :---------------: | ---------: | ----------: | ------------: | ----------: | -------------: |
  | `KoELECTRA-Base`  |        256 |        700K |          2e-4 |         512 |           0.33 |
  | `KoELECTRA-Small` |        512 |        300K |          5e-4 |         512 |            1.0 |

- In case of `KoELECTRA-Small` model, the same options as `ELECTRA-Small++` in the original paper were used.

  - This is the same setting as the small model distributed by the official ELECTRA code.
  - Also, unlike `KoELECTRA-Base`, the model size of Generator and Disciminator is same.

- Except for `Batch size` and `Train steps`, other hyperparameters are same as that of original paper.

  - I tried changing other hyperparameters and running them, but setting them as same as the original paper performed best.

- **TPU v3-8** was used for pretraining, and the base model took **about 7 days** and the small model took **about 3 days**.

  - More detail about using TPU on GCP, see [[Using TPU for Pretraining]](./docs/tpu_training_EN.md)

## KoELECTRA on 🤗 Transformers 🤗

- `ElectraModel` is officially supported from `Transformers v2.8.0`.

- `ElectraModel` is similar to `BertModel` except that it does not return `pooled_output`.

- ELECTRA uses `discriminator` for finetuning.

```python
from transformers import ElectraModel, ElectraTokenizer

# KoELECTRA-Base
model = ElectraModel.from_pretrained("monologg/koelectra-base-discriminator")
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-discriminator")

# KoELECTRA-Small
model = ElectraModel.from_pretrained("monologg/koelectra-small-discriminator")
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-small-discriminator")
```

```python
>>> from transformers import ElectraTokenizer
>>> tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-discriminator")
>>> tokenizer.tokenize("[CLS] 한국어 ELECTRA를 공유합니다. [SEP]")
['[CLS]', '한국어', 'E', '##L', '##EC', '##T', '##RA', '##를', '공유', '##합니다', '.', '[SEP]']
>>> tokenizer.convert_tokens_to_ids(['[CLS]', '한국어', 'E', '##L', '##EC', '##T', '##RA', '##를', '공유', '##합니다', '.', '[SEP]'])
[2, 18429, 41, 6240, 15229, 6204, 20894, 5689, 12622, 10690, 18, 3]
```

## Result on Subtask

**This is the result of running with the config as it is, and if hyperparameter tuning is additionally performed, better performance may come out.**

For code and more detail, see [[Finetuning]](./finetune/README_EN.md)

### Base Model

|                    | Size  | **NSMC**<br/>(acc) | **Naver NER**<br/>(F1) | **PAWS**<br/>(acc) | **KorNLI**<br/>(acc) | **KorSTS**<br/>(spearman) | **Question Pair**<br/>(acc) | **KorQuaD (Dev)**<br/>(EM/F1) |
| :----------------- | :---: | :----------------: | :--------------------: | :----------------: | :------------------: | :-----------------------: | :-------------------------: | :---------------------------: |
| KoBERT             | 351M  |       89.63        |         86.11          |       80.65        |        79.00         |           79.64           |            93.93            |         52.81 / 80.27         |
| XLM-Roberta-Base        | 1.03G |       89.49        |         86.26          |     **82.95**      |        79.92         |           79.09           |            93.53            |         64.70 / 88.94         |
| HanBERT            | 614M  |       90.16        |       **87.31**        |       82.40        |      **80.89**       |         **83.33**         |            94.19            |       **78.74 / 92.02**       |
| **KoELECTRA-Base** | 423M  |     **90.21**      |         86.87          |       81.90        |        80.85         |           83.21           |          **94.20**          |         61.10 / 89.59         |

In case of `KoELECTRA-Base`, it shows better performance than `KoBERT`, and similar performance in `HanBERT` on some tasks.

### Small Model

|                     | Size | **NSMC**<br/>(acc) | **Naver NER**<br/>(F1) | **PAWS**<br/>(acc) | **KorNLI**<br/>(acc) | **KorSTS**<br/>(spearman) | **Question Pair**<br/>(acc) | **KorQuaD (Dev)**<br/>(EM/F1) |
| :------------------ | :--: | :----------------: | :--------------------: | :----------------: | :------------------: | :-----------------------: | :-------------------------: | :---------------------------: |
| DistilKoBERT        | 108M |       88.41        |       **84.13**        |       62.55        |        70.55         |           73.21           |            92.48            |         54.12 / 77.80         |
| **KoELECTRA-Small** | 53M  |     **88.76**      |         84.11          |     **74.15**      |      **76.27**       |         **77.00**         |          **93.01**          |       **58.13 / 86.82**       |

In case of `KoELECTRA-Small`, overall performance is better than `DistilKoBERT`.

## Acknowledgement

KoELECTRA was created with Cloud TPU support from the Tensorflow Research Cloud (TFRC) program.

## Reference

- [ELECTRA](https://github.com/google-research/electra)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [Tensorflow Research Cloud](https://www.tensorflow.org/tfrc?hl=ko)
- [Chinese ELECTRA](https://github.com/ymcui/Chinese-ELECTRA/blob/master/README_EN.md)
