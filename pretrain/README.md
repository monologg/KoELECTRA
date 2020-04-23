# ELECTRA Pretraining

config 폴더에 `base`와 `small`에 사용한 hparams이 있습니다.

직접 사용하시려면 config에서 `tpu_name`, `tpu_zone`, `vocab_size`를 바꿔야 합니다.

## Make tfrecords

```bash
# 우선 `data` 디렉토리를 만든 후, corpus를 여러 개로 분리해 놓습니다.
$ mkdir data
$ split -a 4 -l {$NUM_LINES_PER_FILE} -d {$CORPUS_FILE} ./data/data_
```

```bash
python3 build_pretraining_dataset.py --corpus-dir data \
                                     --vocab-file vocab.txt \
                                     --output-dir pretrain_tfrecords \
                                     --max-seq-length 512 \
                                     --num-processes 4 \
                                     --no-lower-case
```

## How to Run Pretraining

```bash
# Base model
$ python3 run_pretraining.py --data-dir gs://{$BUCKET_NAME} --model-name {$BASE_OUTPUT_DIR} --hparams config/base_config.json

# Small model
$ python3 run_pretraining.py --data-dir gs://{$BUCKET_NAME} --model-name {$SMALL_OUTPUT_DIR} --hparams config/small_config.json
```
