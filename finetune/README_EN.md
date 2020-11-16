[한국어](./README.md) | [English](./README_EN.md)

# Finetuning (Benchmark on subtask)

- Write the code based on [Transformers examples](https://github.com/huggingface/transformers/blob/master/examples/README.md)
- For finetuning, `discriminator` was used
- Write the code based on single GPU
- Apart from the KoELECTRA license, there is a separate license for each dataset

## Updates

**July 7, 2020**

- The code has been modified for `transformers v3` compatibility. Previously, `ElectraForSequenceClassification` was not supported in official transformers library so I implemented it myself. But recently it has been implemented in official library, so I used the official implementation.
- The `ElectraForSequenceClassification` in `src/model.py` and [modeling_electra.py](https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_electra.py) are implemented differently.

## Requirements

```python
torch==1.6.0
transformers==3.3.1
seqeval
fastprogress
attrdict
```

## How to Run

```bash
$ python3 run_seq_cls.py --task {$TASK_NAME} --config_file {$CONFIG_FILE}
```

```bash
$ python3 run_seq_cls.py --task nsmc --config_file koelectra-base.json
$ python3 run_seq_cls.py --task kornli --config_file koelectra-base.json
$ python3 run_seq_cls.py --task paws --config_file koelectra-base.json
$ python3 run_seq_cls.py --task question-pair --config_file koelectra-base-v2.json
$ python3 run_seq_cls.py --task korsts --config_file koelectra-small-v2.json
$ python3 run_ner.py --task naver-ner --config_file koelectra-small.json
$ python3 run_squad.py --task korquad --config_file xlm-roberta.json
```

## Result

### Base Model

|                       | Size  | **NSMC**<br/>(acc) | **Naver NER**<br/>(F1) | **PAWS**<br/>(acc) | **KorNLI**<br/>(acc) | **KorSTS**<br/>(spearman) | **Question Pair**<br/>(acc) | **KorQuaD (Dev)**<br/>(EM/F1) | **Korean-Hate-Speech (Dev)**<br/>(F1) |
| :-------------------- | :---: | :----------------: | :--------------------: | :----------------: | :------------------: | :-----------------------: | :-------------------------: | :---------------------------: | :-----------------------------------: |
| KoBERT                | 351M  |       89.59        |         87.92          |       81.25        |        79.62         |           81.59           |            94.85            |         51.75 / 79.15         |                 66.21                 |
| XLM-Roberta-Base      | 1.03G |       89.03        |         86.65          |       82.80        |        80.23         |           78.45           |            93.80            |         64.70 / 88.94         |                 64.06                 |
| HanBERT               | 614M  |       90.06        |         87.70          |       82.95        |        80.32         |           82.73           |            94.72            |         78.74 / 92.02         |               **68.32**               |
| KoELECTRA-Base        | 423M  |       90.33        |         87.18          |       81.70        |        80.64         |           82.00           |            93.54            |         60.86 / 89.28         |                 66.09                 |
| KoELECTRA-Base-v2     | 423M  |       89.56        |         87.16          |       80.70        |        80.72         |           82.30           |            94.85            |         84.01 / 92.40         |                 67.45                 |
| **KoELECTRA-Base-v3** | 431M  |     **90.63**      |       **88.11**        |     **84.45**      |      **82.24**       |         **85.53**         |          **95.25**          |       **84.83 / 93.45**       |                 67.61                 |

### Small Model

|                        | Size | **NSMC**<br/>(acc) | **Naver NER**<br/>(F1) | **PAWS**<br/>(acc) | **KorNLI**<br/>(acc) | **KorSTS**<br/>(spearman) | **Question Pair**<br/>(acc) | **KorQuaD (Dev)**<br/>(EM/F1) | **Korean-Hate-Speech (Dev)**<br/>(F1) |
| :--------------------- | :--: | :----------------: | :--------------------: | :----------------: | :------------------: | :-----------------------: | :-------------------------: | :---------------------------: | :-----------------------------------: |
| DistilKoBERT           | 108M |       88.60        |         84.65          |       60.50        |        72.00         |           72.59           |            92.48            |         54.40 / 77.97         |                 60.72                 |
| KoELECTRA-Small        | 53M  |       88.83        |         84.38          |       73.10        |        76.45         |           76.56           |            93.01            |         58.04 / 86.76         |                 63.03                 |
| KoELECTRA-Small-v2     | 53M  |       88.83        |         85.00          |       72.35        |        78.14         |           77.84           |            93.27            |         81.43 / 90.46         |                 60.14                 |
| **KoELECTRA-Small-v3** | 54M  |     **89.36**      |       **85.40**        |     **77.45**      |      **78.60**       |         **80.79**         |          **94.85**          |       **82.11 / 91.13**       |               **63.07**               |

## Reference

- [Transformers Examples](https://github.com/huggingface/transformers/blob/master/examples/README.md)
- [NSMC](https://github.com/e9t/nsmc)
- [Naver NER Dataset](https://github.com/naver/nlp-challenge)
- [PAWS](https://github.com/google-research-datasets/paws)
- [KorNLI/KorSTS](https://github.com/kakaobrain/KorNLUDatasets)
- [KorQuad](https://korquad.github.io/category/1.0_KOR.html)
- [Korean Hate Speech](https://github.com/kocohub/korean-hate-speech)
- [KoBERT](https://github.com/SKTBrain/KoBERT)
- [HanBERT](https://github.com/tbai2019/HanBert-54k-N)
- [HanBert Transformers](https://github.com/monologg/HanBert-Transformers)
