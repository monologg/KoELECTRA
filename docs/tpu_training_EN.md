[한국어](./tpu_training.md) | [English](./tpu_training_EN.md)

# How to Use TPU for Pretraining ELECTRA

## 1. Apply for Tensorflow Research Cloud

Tensorflow Research Cloud (TFRC) is a program that allows you to use TPU for free for 1 month.

When you go to the [link](https://www.tensorflow.org/tfrc) and apply, you will receive an email.

![image](https://user-images.githubusercontent.com/28896432/79709907-61a92300-82fe-11ea-9773-9ac63b5ebbb6.png)

If you fill out and submit additional applications as required by the email, you will receive a reply as follows, and from that point, you will be able to use TPU in GCP:)

![image](https://user-images.githubusercontent.com/28896432/79709997-9ddc8380-82fe-11ea-9040-06d8ef9c1f1b.png)

## 2. Upload data on Bucket

- When using TPU, all input files must go through the Cloud storage bucket. ([Related FAQ](https://cloud.google.com/tpu/docs/troubleshooting?hl=en#common-errors))

### 2.1. Create Bucket

- Let's name the bucket `test-for-electra`.

- Go to `[Storage]`-`[Browser]` on the left side of the GCP main page.

- Click 'Create Bucket'.

- It is recommended to make a bucket in the same region as the TPU to be used.

  ![image](https://user-images.githubusercontent.com/28896432/79711012-a84c4c80-8301-11ea-955c-39dc604f5c10.png)

### 2.2. File Upload

- Upload prepared `pretrain_tfrecords` and `vocab.txt` to Bucket.

  ![image](https://user-images.githubusercontent.com/28896432/79739355-0a747400-8339-11ea-8de2-f78f8ade887f.png)

## 3. Create GCP VM & TPU

- Rather than making VM and TPU separately, it is recommended to open the `cloud shell` at the top right and enter the following command.

- It doesn't matter if the VM instance is light because the storage is processed by the bucket and the operation is performed by the TPU.

```bash
$ ctpu up --zone=europe-west4-a --tf-version=1.15 \
          --tpu-size=v3-8 --machine-type=n1-standard-2 \
          --disk-size-gb=20 --name={$VM_NAME}
```

![image](https://user-images.githubusercontent.com/28896432/79740137-24fb1d00-833a-11ea-9be8-e317521fa178.png)

## 4. Now Pretrain your own ELECTRA

```bash
$ git clone https://github.com/google-research/electra
$ cd electra
$ python3 run_pretraining.py --data-dir gs://{$BUCKET_NAME} \
                             --model-name {$MODEL_NAME} \
                             --hparams {$CONFIG_PATH}
```

## 5. Delete Instance and Bucket after completing training

```bash
$ ctpu delete --zone=europe-west4-a --name={$VM_NAME}
$ gsutil rm -r gs://test-for-electra
```

## Reference

- [electra](https://github.com/google-research/electra)
- [A Pipeline Of Pretraining Bert On Google TPU](https://github.com/pren1/A_Pipeline_Of_Pretraining_Bert_On_Google_TPU)
- [Official TPU Documentation](https://cloud.google.com/tpu/docs)
