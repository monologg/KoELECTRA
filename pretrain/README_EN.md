[한국어](./README.md) | [English](./README_EN.md)

# ELECTRA Pretraining

In the config folder, there are hparams used for `base` and `small`.

To use it yourself, you need to change `tpu_name`, `tpu_zone`, and `vocab_size` in config.

## Make tfrecords

```bash
# First, create the `data` directory, then separate the corpus
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
