[한국어](./README.md) | [English](./README_EN.md)

# Finetuning (Benchmark on subtask)

- Write the code based on [Transformers examples](https://github.com/huggingface/transformers/blob/master/examples/README.md)
- For finetuning, `discriminator` was used
- Write the code based on single GPU

## Updates

**July 7, 2020**

- The code has been modified for `transformers v3` compatibility. Previously, `ElectraForSequenceClassification` was not supported in official transformers library so I implemented it myself. But recently it has been implemented in official library, so I used the official implementation.
- The `ElectraForSequenceClassification` in `src/model.py` and [modeling_electra.py](https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_electra.py) are implemented differently.
  - Accuracy of nsmc in [Result](https://github.com/monologg/KoELECTRA/blob/master/finetune/README.md#result) is based on the model of `src/model.py`.

## Requirements

```python
torch==1.5.1
transformers==3.0.2
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

|                       | Size  | **NSMC**<br/>(acc) | **Naver NER**<br/>(F1) | **PAWS**<br/>(acc) | **KorNLI**<br/>(acc) | **KorSTS**<br/>(spearman) | **Question Pair**<br/>(acc) | **KorQuaD (Dev)**<br/>(EM/F1) |
| :-------------------- | :---: | :----------------: | :--------------------: | :----------------: | :------------------: | :-----------------------: | :-------------------------: | :---------------------------: |
| KoBERT                | 351M  |       89.63        |         86.11          |       80.65        |        79.00         |           79.64           |            93.93            |         52.81 / 80.27         |
| XLM-Roberta-Base      | 1.03G |       89.49        |         86.26          |       82.95        |        79.92         |           79.09           |            93.53            |         64.70 / 88.94         |
| HanBERT               | 614M  |       90.16        |       **87.31**        |       82.40        |      **80.89**       |           83.33           |            94.19            |         78.74 / 92.02         |
| **KoELECTRA-Base**    | 423M  |     **90.21**      |         86.87          |       81.90        |        80.85         |           83.21           |            94.20            |         61.10 / 89.59         |
| **KoELECTRA-Base-v2** | 423M  |       89.70        |         87.02          |     **83.90**      |        80.61         |         **84.30**         |          **94.72**          |       **84.34 / 92.58**       |

\*The size of HanBERT = Bert Model + Tokenizer DB

### Small Model

|                        | Size | **NSMC**<br/>(acc) | **Naver NER**<br/>(F1) | **PAWS**<br/>(acc) | **KorNLI**<br/>(acc) | **KorSTS**<br/>(spearman) | **Question Pair**<br/>(acc) | **KorQuaD (Dev)**<br/>(EM/F1) |
| :--------------------- | :--: | :----------------: | :--------------------: | :----------------: | :------------------: | :-----------------------: | :-------------------------: | :---------------------------: |
| DistilKoBERT           | 108M |       88.41        |         84.13          |       62.55        |        70.55         |           73.21           |            92.48            |         54.12 / 77.80         |
| **KoELECTRA-Small**    | 53M  |     **88.76**      |         84.11          |       74.15        |        76.27         |           77.00           |            93.01            |         58.13 / 86.82         |
| **KoELECTRA-Small-v2** | 53M  |       88.64        |       **85.05**        |     **74.50**      |      **76.76**       |         **78.28**         |          **93.66**          |       **81.43 / 90.37**       |

\***This is the result of running with the config as it is, and if hyperparameter tuning is additionally performed, better performance may come out.**

In case of `KoELECTRA-Base`, it shows better performance than `KoBERT`, and similar performance in `HanBERT` on some tasks.

In case of `KoELECTRA-Small`, overall performance is better than `DistilKoBERT`.

## Reference

- [Transformers Examples](https://github.com/huggingface/transformers/blob/master/examples/README.md)
- [NSMC](https://github.com/e9t/nsmc)
- [Naver NER Dataset](https://github.com/naver/nlp-challenge)
- [PAWS](https://github.com/google-research-datasets/paws)
- [KorNLI/KorSTS](https://github.com/kakaobrain/KorNLUDatasets)
- [KorQuad](https://korquad.github.io/category/1.0_KOR.html)
- [KoBERT](https://github.com/SKTBrain/KoBERT)
- [HanBERT](https://github.com/tbai2019/HanBert-54k-N)
- [HanBert Transformers](https://github.com/monologg/HanBert-Transformers)
