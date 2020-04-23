[한국어](./wordpiece_vocab.md) | [English](./wordpiece_vocab_EN.md)

# Wordpiece Vocabulary

The alternatives for Korean Tokenizer are `Sentencepiece`, `Mecab`, and `Wordpiece`.

Since BERT, ELECTRA, etc. basically use `Wordpiece`, the Tokenizer provided by default is also written to be compatible with this. In other words, if you want to use `Sentencepiece` or `Mecab`, you have to create a **customized tokenizer**, which causes inconvenience in using the model directly in the library.

The main purpose of this project was to make it possible to use the library immediately **without creating an additional tokenization file**, and decided to use `Wordpiece`.

## Original wordpiece code is NOT available!

<p float="left" align="left">
    <img width="800" src="https://user-images.githubusercontent.com/28896432/80015023-19f7e680-850c-11ea-90d3-436ca253a7a1.png" />  
</p>

**Wordpiece builder used in official BERT is not provided**. Other alternatives have been presented on the BERT official Github, but the exact same Wordpiece vocab doesn't come out.

Several open sources have implemented the Wordpiece vocab builder, but when the input file is very large, memory and speed issues often occur.

## Huggingface Tokenizers

<p float="left" align="center">
    <img width="600" src="https://user-images.githubusercontent.com/28896432/80016455-1c5b4000-850e-11ea-8432-3c356c11f932.png" />  
</p>

Finally, a wordpiece vocabulary was created using the `Tokenizers` library recently released by Huggingface.

Using this library, even if corpus is very large, there is no memory issue, and it is implemented in Rust, so it is also faster than Python.

## Code for building Wordpiece vocab

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

- It should be noted that when `lowercase=False`, `strip_accent=False` must be used.

- In order to reduce the `[UNK]` ratio as much as possible, it should be trained to cover all characters. (`limit_alphabet`)

- After the vocab was created, 200 `[unused]` tokens were additionally added to the vocab.

- Under the premise that preprocessing was done, the UNK Ratio was much lower compared to the sentencepiece.

- For the vocab, please see [vocab.txt](../pretrain/vocab.txt)

## Reference

- [Sentencepiece vs Wordpiece](https://wikidocs.net/22592)
- [Learning a new WordPiece vocabulary](https://github.com/google-research/bert#learning-a-new-wordpiece-vocabulary)
- [kwonmha's bert-vocab-builder](https://github.com/kwonmha/bert-vocab-builder)
- [Huggingface Tokenizers](https://github.com/huggingface/tokenizers)
