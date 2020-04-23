# Preprocessing

Pretrained Language Model 성능에 가장 큰 영향을 주는 것은 `Corpus의 quality`라고 생각합니다. 그러나 뉴스 등의 데이터를 보면 불필요한 문장들이 많기에 이를 전처리하는 과정이 필요했습니다.

## Criteria

### 1. 한글, 영어, 띄어쓰기, 일부 특수 문자 등을 제외하고 모두 제거

```python
re.compile(r'[^ .,?!/@$%~％·∼()\x00-\x7F가-힣]+')
```

특히 **한자**의 경우 Vocab의 quality가 많이 떨어뜨린다고 판단하여 제외하였습니다

### 2. 해시태그, 이메일, @user 제거

### 3. 일부 Brace로 감싸진 단어 제거

`[]`, `{}`, `【】`, `<>`가 대표적

### 4. 한국어 문장 분리기 사용

```bash
$ pip3 install kss
```

### 5. 문장 분리 후 **일정 길이**를 넘지 못하는 문장은 제외

### 6. 뉴스 관련 문장 제거

뉴스는 Noise가 있는 문장 (ex. `무단전재`, `(서울=뉴스1)`이 포함된 문장)이 너무 많은데, 하나하나 판별하는 것은 너무 어려워서 데이터 손실을 보더라도 Noise일 가능성이 있으면 무조건 제외하였습니다.

### 7. 반복되는 글자 제거

```python
from soynlp.normalizer import repeat_normalize

repeat_normalize('와하하하하하하하하하핫', num_repeats=2) # '와하하핫'
```

### 8. 최종적으로 여러 개의 문장이 나오면, 그 중에서 중복되는 문장은 제거

## Reference

- [kor-text-preprocess](https://github.com/YongWookHa/kor-text-preprocess)
- [Korean Sentence Splitter](https://github.com/likejazz/korean-sentence-splitter)
- [soynlp](https://github.com/lovit/soynlp)
