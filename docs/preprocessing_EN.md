[한국어](./preprocessing.md) | [English](./preprocessing_EN.md)

# Preprocessing

I think that `corpus quality` is the most important for the performance of the Pretrained Language Model. However, when looking at the data such as news, there were many unnecessary sentences, so it was necessary to preprocess them.

## Criteria

### 1. Remove all but Korean, English, spaces, and some special characters

```python
re.compile(r'[^ .,?!/@$%~％·∼()\x00-\x7F가-힣]+')
```

Particularly, in the case of **chinese characters**, it was judged that the quality of the vocab dropped significantly and was excluded.

### 2. Remove hashtag, email, @user

### 3. Remove some parenthesized words

For example, `[]`, `{}`, `【】`, `<>`

### 4. Use Korean Sentence Splitter

```bash
$ pip3 install kss
```

### 5. After separating sentences, sentences that do not exceed the certain length are excluded.

### 6. Remove news-related sentences

There are too many sentences with noise (ex. `무단전재`, `(서울=뉴스1)`) in news corpus. It is too difficult to judge one by one, so if there is a possibility of noise even if there will be some data loss, it is unconditionally excluded.

### 7. Remove repeated characters

```python
from soynlp.normalizer import repeat_normalize

repeat_normalize('와하하하하하하하하하핫', num_repeats=2) # '와하하핫'
```

### 8. Remove duplicate sentences

## Reference

- [kor-text-preprocess](https://github.com/YongWookHa/kor-text-preprocess)
- [Korean Sentence Splitter](https://github.com/likejazz/korean-sentence-splitter)
- [soynlp](https://github.com/lovit/soynlp)
