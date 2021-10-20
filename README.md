[한국어](./README.md) | [English](./README_EN.md)

# KoELECTRA

<p float="left" align="center">
    <img width="900" src="https://user-images.githubusercontent.com/28896432/80024445-0f444e00-851a-11ea-9137-9da2abfd553d.png" />  
</p>

[ELECTRA](https://openreview.net/pdf?id=r1xMH1BtvB)는 `Replaced Token Detection`, 즉 generator에서 나온 token을 보고 discriminator에서 "real" token인지 "fake" token인지 판별하는 방법으로 학습을 합니다. 이 방법은 모든 input token에 대해 학습할 수 있다는 장점을 가지며, BERT 등과 비교했을 때 더 좋은 성능을 보였습니다.

KoELECTRA는 **34GB의 한국어 text**로 학습하였고, 이를 통해 나온 `KoELECTRA-Base`와 `KoELECTRA-Small` 두 가지 모델을 배포하게 되었습니다.

또한 KoELECTRA는 **Wordpiece 사용**, **모델 s3 업로드** 등을 통해 OS 상관없이 `Transformers` 라이브러리만 설치하면 곧바로 사용할 수 있습니다.

## Updates

**April 27, 2020**

- 2개의 Subtask (`KorSTS`, `QuestionPair`)에 대해 추가적으로 finetuning을 진행하였고, 기존 5개의 Subtask에 대해서도 결과를 업데이트하였습니다.

**June 3, 2020**

- [EnlipleAI PLM](https://github.com/enlipleai/kor_pratrain_LM)에서 사용된 vocabulary를 이용하여 `KoELECTRA-v2`를 제작하였습니다. Base 모델과 Small 모델 모두 `KorQuaD`에서 성능 향상을 보였습니다.

**October 9, 2020**

- `모두의 말뭉치`를 추가적으로 사용하여 `KoELECTRA-v3`를 제작하였습니다. Vocab도 `Mecab`과 `Wordpiece`를 이용하여 새로 제작하였습니다.
- `Huggingface Transformers`의 `ElectraForSequenceClassification` 공식 지원 등을 고려하여 기존 Subtask 결과를 새로 Update하였습니다. 또한 [Korean-Hate-Speech](https://github.com/kocohub/korean-hate-speech)의 결과도 추가했습니다.

```python
from transformers import ElectraModel, ElectraTokenizer

model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
```

**May 26, 2021**

- `torch<=1.4` 에서 로딩이 되지 않는 이슈 해결 (모델 수정 후 재업로드 완료) ([Related Issue](https://github.com/pytorch/pytorch/issues/48915))
- huggingface hub에 `tensorflow v2` 모델 업로드 (`tf_model.h5`)

**Oct 20, 2021**

- `tf_model.h5`에서 바로 로딩하는 부분이 여러 이슈가 존재하여 제거 (`from_pt=True`로 로딩하는 것으로 되돌림)

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

- 이번 프로젝트의 가장 큰 목적은 **Transformers 라이브러리만 있으면 모델을 곧바로 사용 가능하게 만드는 것**이었고, 이에 Sentencepiece, Mecab을 사용하지 않고 원 논문과 코드에서 사용한 `Wordpiece`를 사용하였습니다.
- 자세한 내용은 [[Wordpiece Vocabulary]](./docs/wordpiece_vocab.md) 참고

|     | Vocab Len | do_lower_case |
| --- | --------: | ------------: |
| v1  |     32200 |         False |
| v2  |     32200 |         False |
| v3  |     35000 |         False |

### Data

- `v1`, `v2`의 경우 **약 14G Corpus** (2.6B tokens)를 사용했습니다. (뉴스, 위키, 나무위키)
- `v3`의 경우 **약 20G의 모두의 말뭉치**를 추가적으로 사용했습니다. (신문, 문어, 구어, 메신저, 웹)

### Pretraining Details

| Model        | Batch Size | Train Steps |   LR | Max Seq Len | Generator Size | Train Time |
| :----------- | ---------: | ----------: | ---: | ----------: | -------------: | ---------: |
| `Base v1,2`  |        256 |        700K | 2e-4 |         512 |           0.33 |         7d |
| `Base v3`    |        256 |        1.5M | 2e-4 |         512 |           0.33 |        14d |
| `Small v1,2` |        512 |        300K | 5e-4 |         512 |            1.0 |         3d |
| `Small v3`   |        512 |        800K | 5e-4 |         512 |            1.0 |         7d |

- `KoELECTRA-Small` 모델의 경우 원 논문에서의 `ELECTRA-Small++`와 **동일한 옵션**을 사용하였습니다.

  - 이는 공식 ELECTRA에서 배포한 Small 모델과 설정이 동일합니다.
  - 또한 `KoELECTRA-Base`와는 달리, Generator와 Discriminator의 모델 사이즈(=`generator_hidden_size`)가 동일합니다.

- `Batch size`와 `Train steps`을 제외하고는 **원 논문의 Hyperparameter와 동일**하게 가져갔습니다.

  - 다른 hyperparameter를 변경하여 돌려봤지만 원 논문과 동일하게 가져간 것이 성능이 가장 좋았습니다.

- **TPU v3-8**을 이용하여 학습하였고, GCP에서의 TPU 사용법은 [[Using TPU for Pretraining]](./docs/tpu_training.md)에 정리하였습니다.

## KoELECTRA on 🤗 Transformers 🤗

- `Transformers v2.8.0`부터 `ElectraModel`을 공식 지원합니다.

- **Huggingface S3**에 모델이 이미 업로드되어 있어서, **모델을 직접 다운로드할 필요 없이** 곧바로 사용할 수 있습니다.

- `ElectraModel`은 `pooled_output`을 리턴하지 않는 것을 제외하고 `BertModel`과 유사합니다.

- ELECTRA는 finetuning시에 `discriminator`를 사용합니다.

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

**config의 세팅을 그대로 하여 돌린 결과이며, hyperparameter tuning을 추가적으로 할 시 더 좋은 성능이 나올 수 있습니다.**

코드 및 자세한 내용은 [[Finetuning]](./finetune/README.md) 참고

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

KoELECTRA은 **Tensorflow Research Cloud (TFRC)** 프로그램의 Cloud TPU 지원으로 제작되었습니다. 또한 `KoELECTRA-v3`는 **모두의 말뭉치**의 도움으로 제작되었습니다.

## Citation

이 코드를 연구용으로 사용하는 경우 아래와 같이 인용해주세요.

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
- [모두의 말뭉치](https://corpus.korean.go.kr/)
