[한국어](./README.md) | [English](./README_EN.md)

# KoELECTRA

<p float="left" align="center">
    <img width="900" src="https://user-images.githubusercontent.com/28896432/80024445-0f444e00-851a-11ea-9137-9da2abfd553d.png" />  
</p>

[ELECTRA](https://openreview.net/pdf?id=r1xMH1BtvB)는 `Replaced Token Detection`, 즉 generator에서 나온 token을 보고 discriminator에서 "real" token인지 "fake" token인지 판별하는 방법으로 학습을 합니다. 이 방법은 모든 input token에 대해 학습할 수 있다는 장점을 가지며, BERT 등과 비교했을 때 더 좋은 성능을 보였습니다.

KoELECTRA는 **14GB의 한국어 text** (96M sentences, 2.6B tokens)로 학습하였고, 이를 통해 나온 `KoELECTRA-Base`와 `KoELECTRA-Small` 두 가지 모델을 배포하게 되었습니다.

또한 KoELECTRA는 **Wordpiece 사용**, **모델 s3 업로드** 등을 통해 OS 상관없이 `Transformers` 라이브러리만 설치하면 곧바로 사용할 수 있습니다.

## Updates

**April 27, 2020** - 2개의 Subtask (`KorSTS`, `QuestionPair`)에 대해 추가적으로 finetuning을 진행하였고, 기존 5개의 Subtask에 대해서도 결과를 업데이트하였습니다.

## About KoELECTRA

|                   |               | Layers | Embedding Size | Hidden Size | # heads | Size |
| ----------------- | ------------: | -----: | -------------: | ----------: | ------: | ---: |
| `KoELECTRA-Base`  | Discriminator |     12 |            768 |         768 |      12 | 423M |
|                   |     Generator |     12 |            768 |         256 |       4 | 134M |
| `KoELECTRA-Small` | Discriminator |     12 |            128 |         256 |       4 |  53M |
|                   |     Generator |     12 |            128 |         256 |       4 |  53M |

### Vocabulary

이번 프로젝트의 가장 큰 목적은 **Transformers 라이브러리만 있으면 모델을 곧바로 사용 가능하게 만드는 것**이었고, 이에 Sentencepiece, Mecab을 사용하지 않고 원 논문과 코드에서 사용한 `Wordpiece`를 사용하였습니다.

- Vocab의 사이즈는 `32200`개로 `[unused]` 토큰 200개를 추가하였습니다.
- Cased (`do_lower_case=False`)로 처리하였습니다.

자세한 내용은 [[Wordpiece Vocabulary]](./docs/wordpiece_vocab.md) 참고

### Pretraining Details

- Data의 경우 전처리가 완료된 **14G의 Corpus**(2.6B tokens)를 사용하였습니다. (전처리 관련 내용은 [[Preprocessing]](./docs/preprocessing.md) 참고)

  |       Model       | Batch Size | Train Steps | Learning Rate | Max Seq Len | Generator Size |
  | :---------------: | ---------: | ----------: | ------------: | ----------: | -------------: |
  | `KoELECTRA-Base`  |        256 |        700K |          2e-4 |         512 |           0.33 |
  | `KoELECTRA-Small` |        512 |        300K |          5e-4 |         512 |            1.0 |

- `KoELECTRA-Small` 모델의 경우 원 논문에서의 `ELECTRA-Small++`와 **동일한 옵션**을 사용하였습니다.

  - 이는 공식 ELECTRA에서 배포한 Small 모델과 설정이 동일합니다.
  - 또한 `KoELECTRA-Base`와는 달리, Generator와 Discriminator의 모델 사이즈(=`generator_hidden_size`)가 동일합니다.

- `Batch size`와 `Train steps`을 제외하고는 **원 논문의 Hyperparameter와 동일**하게 가져갔습니다.

  - 다른 hyperparameter를 변경하여 돌려봤지만 원 논문과 동일하게 가져간 것이 성능이 가장 좋았습니다.

- **TPU v3-8**을 이용하여 학습하였고, Base 모델은 **약 7일**, Small 모델은 **약 3일**이 소요되었습니다.

  - GCP에서의 TPU 사용법은 [[Using TPU for Pretraining]](./docs/tpu_training.md)에 정리하였습니다.

## KoELECTRA on 🤗 Transformers 🤗

- `Transformers v2.8.0`부터 `ElectraModel`을 공식 지원합니다.

- **Huggingface S3**에 모델이 이미 업로드되어 있어서, **모델을 직접 다운로드할 필요 없이** 곧바로 사용할 수 있습니다.

- `ElectraModel`은 `pooled_output`을 리턴하지 않는 것을 제외하고 `BertModel`과 유사합니다.

- ELECTRA는 finetuning시에 `discriminator`를 사용합니다.

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

**config의 세팅을 그대로 하여 돌린 결과이며, hyperparameter tuning을 추가적으로 할 시 더 좋은 성능이 나올 수 있습니다.**

코드 및 자세한 내용은 [[Finetuning]](./finetune/README.md) 참고

### Base Model

|                    | Size  | **NSMC**<br/>(acc) | **Naver NER**<br/>(F1) | **PAWS**<br/>(acc) | **KorNLI**<br/>(acc) | **KorSTS**<br/>(spearman) | **Question Pair**<br/>(acc) | **KorQuaD (Dev)**<br/>(EM/F1) |
| :----------------- | :---: | :----------------: | :--------------------: | :----------------: | :------------------: | :-----------------------: | :-------------------------: | :---------------------------: |
| KoBERT             | 351M  |       89.63        |         86.11          |       80.65        |        79.00         |           79.64           |            93.93            |         52.81 / 80.27         |
| XLM-Roberta-Base        | 1.03G |       89.49        |         86.26          |     **82.95**      |        79.92         |           79.09           |            93.53            |         64.70 / 88.94         |
| HanBERT            | 614M  |       90.16        |       **87.31**        |       82.40        |      **80.89**       |         **83.33**         |            94.19            |       **78.74 / 92.02**       |
| **KoELECTRA-Base** | 423M  |     **90.21**      |         86.87          |       81.90        |        80.85         |           83.21           |          **94.20**          |         61.10 / 89.59         |

`KoELECTRA-Base`의 경우 `KoBERT`보다 좋은 성능을 보이며, `HanBERT`와 일부 Task에서 유사한 성능을 보입니다.

### Small Model

|                     | Size | **NSMC**<br/>(acc) | **Naver NER**<br/>(F1) | **PAWS**<br/>(acc) | **KorNLI**<br/>(acc) | **KorSTS**<br/>(spearman) | **Question Pair**<br/>(acc) | **KorQuaD (Dev)**<br/>(EM/F1) |
| :------------------ | :--: | :----------------: | :--------------------: | :----------------: | :------------------: | :-----------------------: | :-------------------------: | :---------------------------: |
| DistilKoBERT        | 108M |       88.41        |       **84.13**        |       62.55        |        70.55         |           73.21           |            92.48            |         54.12 / 77.80         |
| **KoELECTRA-Small** | 53M  |     **88.76**      |         84.11          |     **74.15**      |      **76.27**       |         **77.00**         |          **93.01**          |       **58.13 / 86.82**       |

`KoELECTRA-Small`의 경우 전반적으로 `DistilKoBERT`보다 좋은 성능을 보입니다.

## Acknowledgement

KoELECTRA은 Tensorflow Research Cloud (TFRC) 프로그램의 Cloud TPU 지원으로 제작되었습니다.

## Reference

- [ELECTRA](https://github.com/google-research/electra)
- [Huggingface Transformers](https://github.com/huggingface/transformers)
- [Tensorflow Research Cloud](https://www.tensorflow.org/tfrc?hl=ko)
- [Chinese ELECTRA](https://github.com/ymcui/Chinese-ELECTRA/blob/master/README_EN.md)
