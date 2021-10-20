from transformers import (
    AutoModel,
    AutoTokenizer,
    ElectraModel,
    ElectraTokenizer,
    ElectraTokenizerFast,
    TFAutoModel,
    TFElectraModel,
)
from transformers.utils import check_min_version
from transformers.utils.versions import require_version

# NOTE check library version
require_version("torch>=1.4")
require_version("tensorflow>=2.0.0")
check_min_version("4.11.0")


ALL_MODEL_NAME_OR_PATH_LST = [
    "monologg/koelectra-base-discriminator",
    "monologg/koelectra-base-generator",
    "monologg/koelectra-base-v2-discriminator",
    "monologg/koelectra-base-v2-generator",
    "monologg/koelectra-base-v3-discriminator",
    "monologg/koelectra-base-v3-generator",
]


def test_load_auto_pt_model():
    for model_name_or_path in ALL_MODEL_NAME_OR_PATH_LST:
        AutoModel.from_pretrained(model_name_or_path)


def test_load_auto_tf_model():
    for model_name_or_path in ALL_MODEL_NAME_OR_PATH_LST:
        TFAutoModel.from_pretrained(model_name_or_path, from_pt=True)


def test_load_pt_model():
    for model_name_or_path in ALL_MODEL_NAME_OR_PATH_LST:
        ElectraModel.from_pretrained(model_name_or_path)


def test_load_tf_model():
    for model_name_or_path in ALL_MODEL_NAME_OR_PATH_LST:
        TFElectraModel.from_pretrained(model_name_or_path, from_pt=True)


def test_load_auto_tokenizer():
    # Load fast tokenizer
    for model_name_or_path in ALL_MODEL_NAME_OR_PATH_LST:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        assert type(tokenizer) == ElectraTokenizerFast

    # Load slow tokenizer
    for model_name_or_path in ALL_MODEL_NAME_OR_PATH_LST:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
        assert type(tokenizer) == ElectraTokenizer
