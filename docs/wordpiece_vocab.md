[한국어](./wordpiece_vocab.md) | [English](./wordpiece_vocab_EN.md)

# Wordpiece Vocabulary

한국어 Tokenizer의 대안으로는 크게 `Sentencepiece`, `Mecab`, `Wordpiece`가 있습니다.

BERT, ELECTRA 등은 기본적으로 `Wordpiece`를 사용하기에 기본적으로 제공되는 Tokenizer 역시 이에 호환되게 코드가 작성되었습니다. 즉, `Sentencepiece`나 `Mecab`을 사용하려면 **별도의 Tokenizer**를 직접 만들어야 하고, 이렇게 되면 라이브러리에서 모델을 곧바로 사용하는데 불편함이 생깁니다.

이번 프로젝트의 가장 큰 목적은 **추가적인 tokenization 파일을 만들지 않고 곧바로 라이브러리를 사용할 수 있게 하는 것**이었고, 이에 `Wordpiece`를 사용하는 것으로 정했습니다.

## Original wordpiece code is NOT available!

<p float="left" align="left">
    <img width="800" src="https://user-images.githubusercontent.com/28896432/80015023-19f7e680-850c-11ea-90d3-436ca253a7a1.png" />  
</p>

**공식 BERT에서 사용된 Wordpiece Builder는 제공되지 않고 있습니다**. BERT 공식 Github에서 다른 대안들을 제시해줬지만, 완전히 동일한 Wordpiece Vocab이 나오지 않습니다.

몇몇 오픈소스들이 Wordpiece vocab builder 구현을 구현하였지만 **input_file이 매우 클 시 메모리, 속도 등의 이슈**가 종종 발생하였습니다.

## Huggingface Tokenizers

<p float="left" align="center">
    <img width="600" src="https://user-images.githubusercontent.com/28896432/80016455-1c5b4000-850e-11ea-8432-3c356c11f932.png" />  
</p>

최종적으로, 최근 Huggingface에서 발표한 `Tokenizers` 라이브러리를 이용하여 Wordpiece Vocabulary를 만들었습니다.

해당 라이브러리를 사용하면 Corpus가 매우 커도 메모리 이슈가 발생하지 않으며, Rust로 구현이 되어있어 속도 또한 Python보다 빠릅니다.

## Code for building Wordpiece vocab

(`tokenizers==0.7.0`을 기준으로 작성했습니다. 라이브러리가 업데이트되면 API가 바뀔 수 있습니다.)

```python
import os
import argparse
from tokenizers import BertWordPieceTokenizer

parser = argparse.ArgumentParser()

parser.add_argument("--corpus_file", type=str)
parser.add_argument("--vocab_size", type=int, default=32000)
parser.add_argument("--limit_alphabet", type=int, default=6000)

args = parser.parse_args()

tokenizer = BertWordPieceTokenizer(
    vocab_file=None,
    clean_text=True,
    handle_chinese_chars=True,
    strip_accents=False, # Must be False if cased model
    lowercase=False,
    wordpieces_prefix="##"
)

tokenizer.train(
    files=[args.corpus_file],
    limit_alphabet=args.limit_alphabet,
    vocab_size=args.vocab_size
)

tokenizer.save("./", "ch-{}-wpm-{}".format(args.limit_alphabet, args.vocab_size))
```

- 주의해야할 점은 `lowercase=False`로 할 시 `strip_accent=False`로 해줘야 한다는 것입니다.

- `[UNK]`의 비중을 최대한 줄이기 위해 모든 character를 커버할 수 있도록 처리하였습니다. (`limit_alphabet`)

- vocab을 다 만든 후, `[unused]`token 200개를 vocab에 추가적으로 붙였습니다.

- Corpus의 전처리가 완료되었다는 전제하에 sentencepiece와 비교했을 때 UNK Ratio가 훨씬 낮았습니다.

- 완성된 vocab은 [vocab.txt](../pretrain/vocab.txt)을 참고하시면 됩니다.

## Reference

- [Sentencepiece vs Wordpiece](https://wikidocs.net/22592)
- [Learning a new WordPiece vocabulary](https://github.com/google-research/bert#learning-a-new-wordpiece-vocabulary)
- [kwonmha's bert-vocab-builder](https://github.com/kwonmha/bert-vocab-builder)
- [Huggingface Tokenizers](https://github.com/huggingface/tokenizers)
