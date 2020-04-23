# How to Use TPU for Pretraining ELECTRA

## 1. Tensorflow Research Cloud 신청

Tensorflow Research Cloud (TFRC)는 1달 동안 TPU를 무료로 사용할 수 있게 해주는 프로그램입니다.

해당 [링크](https://www.tensorflow.org/tfrc?hl=ko)로 가서 신청을 하게 되면 메일이 하나 오게 됩니다.

![image](https://user-images.githubusercontent.com/28896432/79709907-61a92300-82fe-11ea-9773-9ac63b5ebbb6.png)

해당 메일에서 요구하는 대로 신청서를 추가적으로 작성 후 제출하면 얼마 후 아래와 같이 답장이 오게 되고, 그 때부터 GCP에서 TPU를 사용할 수 있게 됩니다:)

![image](https://user-images.githubusercontent.com/28896432/79709997-9ddc8380-82fe-11ea-9040-06d8ef9c1f1b.png)

## 2. Bucket에 Data 업로드

- TPU를 쓰는 경우 모든 input file을 Cloud storage bucket을 통해야만 합니다. ([관련 FAQ](https://cloud.google.com/tpu/docs/troubleshooting?hl=ko#common-errors))

### 2.1. Bucket 생성

- Bucket의 이름을 `test-for-electra`로 만들어 보겠습니다.

- GCP 메인 페이지 좌측의 `[Storage]` - `[브라우저]` 로 이동

- `버킷 만들기` 클릭

- 사용할 TPU와 동일한 Region에 Bucket 만드는 것을 권장

  ![image](https://user-images.githubusercontent.com/28896432/79711012-a84c4c80-8301-11ea-955c-39dc604f5c10.png)

### 2.2. File Upload

- 준비한 `pretrain_tfrecords`와 `vocab.txt`를 Bucket에 업로드

  ![image](https://user-images.githubusercontent.com/28896432/79739355-0a747400-8339-11ea-8de2-f78f8ade887f.png)

## 3. GCP VM & TPU 생성

- VM과 TPU를 각각 따로 만드는 것보다, 우측 상단의 `cloud shell`을 열어 아래의 명령어를 입력하는 것을 추천합니다.

- 저장소는 Bucket이, 연산은 TPU에서 처리하기 때문에 VM Instance는 가벼운 것을 써도 상관이 없습니다.

```bash
$ ctpu up --zone=europe-west4-a --tf-version=1.15 \
          --tpu-size=v3-8 --machine-type=n1-standard-2 \
          --disk-size-gb=20 --name={$VM_NAME}
```

![image](https://user-images.githubusercontent.com/28896432/79740137-24fb1d00-833a-11ea-9be8-e317521fa178.png)

## 4. Electra 학습 진행

```bash
$ git clone https://github.com/google-research/electra
$ cd electra
$ python3 run_pretraining.py --data-dir gs://{$BUCKET_NAME} \
                             --model-name {$MODEL_NAME} \
                             --hparams {$CONFIG_PATH}
```

## 5. 학습 완료 후 Instance, Bucket 삭제

```bash
$ ctpu delete --zone=europe-west4-a --name={$VM_NAME}
$ gsutil rm -r gs://test-for-electra
```

## Reference

- [electra](https://github.com/google-research/electra)
- [A Pipeline Of Pretraining Bert On Google TPU](https://github.com/pren1/A_Pipeline_Of_Pretraining_Bert_On_Google_TPU)
- [Official TPU Documentation](https://cloud.google.com/tpu/docs)
