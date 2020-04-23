import os
import random
import logging

import torch
import numpy as np

from seqeval.metrics import precision_score, recall_score, f1_score, classification_report

from src import (
    KoBertTokenizer,
    HanBertTokenizer,
    ElectraForSequenceClassification,
    ElectraForQuestionAnswering,
    XLMRobertaForQuestionAnswering
)
from transformers import (
    BertConfig,
    DistilBertConfig,
    ElectraConfig,
    XLMRobertaConfig,
    ElectraTokenizer,
    XLMRobertaTokenizer,
    BertForSequenceClassification,
    DistilBertForSequenceClassification,
    XLMRobertaForSequenceClassification,
    BertForTokenClassification,
    DistilBertForTokenClassification,
    XLMRobertaForTokenClassification,
    ElectraForTokenClassification,
    BertForQuestionAnswering,
    DistilBertForQuestionAnswering
)

CONFIG_CLASSES = {
    "kobert": BertConfig,
    "distilkobert": DistilBertConfig,
    "hanbert": BertConfig,
    "koelectra-base": ElectraConfig,
    "koelectra-small": ElectraConfig,
    "xlm-roberta": XLMRobertaConfig
}

TOKENIZER_CLASSES = {
    "kobert": KoBertTokenizer,
    "distilkobert": KoBertTokenizer,
    "hanbert": HanBertTokenizer,
    "koelectra-base": ElectraTokenizer,
    "koelectra-small": ElectraTokenizer,
    "xlm-roberta": XLMRobertaTokenizer
}

MODEL_FOR_SEQUENCE_CLASSIFICATION = {
    "kobert": BertForSequenceClassification,
    "distilkobert": DistilBertForSequenceClassification,
    "hanbert": BertForSequenceClassification,
    "koelectra-base": ElectraForSequenceClassification,
    "koelectra-small": ElectraForSequenceClassification,
    "xlm-roberta": XLMRobertaForSequenceClassification
}

MODEL_FOR_TOKEN_CLASSIFICATION = {
    "kobert": BertForTokenClassification,
    "distilkobert": DistilBertForTokenClassification,
    "hanbert": BertForTokenClassification,
    "koelectra-base": ElectraForTokenClassification,
    "koelectra-small": ElectraForTokenClassification,
    "xlm-roberta": XLMRobertaForTokenClassification
}

MODEL_FOR_QUESTION_ANSWERING = {
    "kobert": BertForQuestionAnswering,
    "distilkobert": DistilBertForQuestionAnswering,
    "hanbert": BertForQuestionAnswering,
    "koelectra-base": ElectraForQuestionAnswering,
    "koelectra-small": ElectraForQuestionAnswering,
    "xlm-roberta": XLMRobertaForQuestionAnswering
}


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def simple_accuracy(labels, preds):
    return (labels == preds).mean()


def acc_score(labels, preds):
    return {
        "acc": simple_accuracy(labels, preds),
    }


def f1_pre_rec(labels, preds):
    return {
        "precision": precision_score(labels, preds, suffix=True),
        "recall": recall_score(labels, preds, suffix=True),
        "f1": f1_score(labels, preds, suffix=True)
    }


def show_ner_report(labels, preds):
    return classification_report(labels, preds, suffix=True)
