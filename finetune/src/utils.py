import random
import logging
from numpy.lib.function_base import average

import torch
import numpy as np

from scipy.stats import pearsonr, spearmanr
from seqeval import metrics as seqeval_metrics
from sklearn import metrics as sklearn_metrics

from src import KoBertTokenizer, HanBertTokenizer
from transformers import (
    BertConfig,
    DistilBertConfig,
    ElectraConfig,
    XLMRobertaConfig,
    ElectraTokenizer,
    XLMRobertaTokenizer,
    BertForSequenceClassification,
    DistilBertForSequenceClassification,
    ElectraForSequenceClassification,
    XLMRobertaForSequenceClassification,
    BertForTokenClassification,
    DistilBertForTokenClassification,
    ElectraForTokenClassification,
    XLMRobertaForTokenClassification,
    BertForQuestionAnswering,
    DistilBertForQuestionAnswering,
    ElectraForQuestionAnswering,
    XLMRobertaForQuestionAnswering,
)

CONFIG_CLASSES = {
    "kobert": BertConfig,
    "distilkobert": DistilBertConfig,
    "hanbert": BertConfig,
    "koelectra-base": ElectraConfig,
    "koelectra-small": ElectraConfig,
    "koelectra-base-v2": ElectraConfig,
    "koelectra-base-v3": ElectraConfig,
    "koelectra-small-v2": ElectraConfig,
    "koelectra-small-v3": ElectraConfig,
    "xlm-roberta": XLMRobertaConfig,
}

TOKENIZER_CLASSES = {
    "kobert": KoBertTokenizer,
    "distilkobert": KoBertTokenizer,
    "hanbert": HanBertTokenizer,
    "koelectra-base": ElectraTokenizer,
    "koelectra-small": ElectraTokenizer,
    "koelectra-base-v2": ElectraTokenizer,
    "koelectra-base-v3": ElectraTokenizer,
    "koelectra-small-v2": ElectraTokenizer,
    "koelectra-small-v3": ElectraTokenizer,
    "xlm-roberta": XLMRobertaTokenizer,
}

MODEL_FOR_SEQUENCE_CLASSIFICATION = {
    "kobert": BertForSequenceClassification,
    "distilkobert": DistilBertForSequenceClassification,
    "hanbert": BertForSequenceClassification,
    "koelectra-base": ElectraForSequenceClassification,
    "koelectra-small": ElectraForSequenceClassification,
    "koelectra-base-v2": ElectraForSequenceClassification,
    "koelectra-base-v3": ElectraForSequenceClassification,
    "koelectra-small-v2": ElectraForSequenceClassification,
    "koelectra-small-v3": ElectraForSequenceClassification,
    "xlm-roberta": XLMRobertaForSequenceClassification,
}

MODEL_FOR_TOKEN_CLASSIFICATION = {
    "kobert": BertForTokenClassification,
    "distilkobert": DistilBertForTokenClassification,
    "hanbert": BertForTokenClassification,
    "koelectra-base": ElectraForTokenClassification,
    "koelectra-small": ElectraForTokenClassification,
    "koelectra-base-v2": ElectraForTokenClassification,
    "koelectra-base-v3": ElectraForTokenClassification,
    "koelectra-small-v2": ElectraForTokenClassification,
    "koelectra-small-v3": ElectraForTokenClassification,
    "koelectra-small-v3-51000": ElectraForTokenClassification,
    "xlm-roberta": XLMRobertaForTokenClassification,
}

MODEL_FOR_QUESTION_ANSWERING = {
    "kobert": BertForQuestionAnswering,
    "distilkobert": DistilBertForQuestionAnswering,
    "hanbert": BertForQuestionAnswering,
    "koelectra-base": ElectraForQuestionAnswering,
    "koelectra-small": ElectraForQuestionAnswering,
    "koelectra-base-v2": ElectraForQuestionAnswering,
    "koelectra-base-v3": ElectraForQuestionAnswering,
    "koelectra-small-v2": ElectraForQuestionAnswering,
    "koelectra-small-v3": ElectraForQuestionAnswering,
    "xlm-roberta": XLMRobertaForQuestionAnswering,
}


def init_logger():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )


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


def pearson_and_spearman(labels, preds):
    pearson_corr = pearsonr(preds, labels)[0]
    spearman_corr = spearmanr(preds, labels)[0]
    return {
        "pearson": pearson_corr,
        "spearmanr": spearman_corr,
        "corr": (pearson_corr + spearman_corr) / 2,
    }


def f1_pre_rec(labels, preds, is_ner=True):
    if is_ner:
        return {
            "precision": seqeval_metrics.precision_score(labels, preds, suffix=True),
            "recall": seqeval_metrics.recall_score(labels, preds, suffix=True),
            "f1": seqeval_metrics.f1_score(labels, preds, suffix=True),
        }
    else:
        return {
            "precision": sklearn_metrics.precision_score(labels, preds, average="macro"),
            "recall": sklearn_metrics.recall_score(labels, preds, average="macro"),
            "f1": sklearn_metrics.f1_score(labels, preds, average="macro"),
        }


def show_ner_report(labels, preds):
    return seqeval_metrics.classification_report(labels, preds, suffix=True)


def compute_metrics(task_name, labels, preds):
    assert len(preds) == len(labels)
    if task_name == "kornli":
        return acc_score(labels, preds)
    elif task_name == "nsmc":
        return acc_score(labels, preds)
    elif task_name == "paws":
        return acc_score(labels, preds)
    elif task_name == "korsts":
        return pearson_and_spearman(labels, preds)
    elif task_name == "question-pair":
        return acc_score(labels, preds)
    elif task_name == "naver-ner":
        return f1_pre_rec(labels, preds, is_ner=True)
    elif task_name == "hate-speech":
        return f1_pre_rec(labels, preds, is_ner=False)
    else:
        raise KeyError(task_name)
