import argparse
import json
import logging
import os
import glob

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.nn import CrossEntropyLoss
from fastprogress.fastprogress import master_bar, progress_bar
from attrdict import AttrDict

from transformers import (
    AdamW,
    get_linear_schedule_with_warmup
)

from src import (
    CONFIG_CLASSES,
    TOKENIZER_CLASSES,
    MODEL_FOR_TOKEN_CLASSIFICATION,
    init_logger,
    set_seed,
    compute_metrics,
    show_ner_report
)

from processor import ner_load_and_cache_examples as load_and_cache_examples
from processor import ner_tasks_num_labels as tasks_num_labels
from processor import ner_processors as processors

logger = logging.getLogger(__name__)

def get_sentence_list(data_dir,data_file_name):
    data_file = os.path.join(data_dir,data_file_name)

    with open(data_file,'r') as f_r:
        if data_file.endswith('.json'):
            data = json.load(f_r)
            sentences = [q['Question'] for q in data]
        else:
            data = f_r.read()
            sentences = data.split('\n')

    return sentences

def test(args, model, tokenizer, labels, test_sentences, mode, global_step=None):
    results = []

    for s in test_sentences:

        result = dict('Question',s)

        inputs = tokenizer(s,return_tensors='pt')

        with torch.no_grad():
            entities = model(**inputs)[0][0].cpu().numpy()
            input_ids = inputs['input_ids'].numpy()[0]

        score = np.exp(entities) / np.exp(entities).sum(-1, keepdims=True)
        labels_idx = score.argmax(axis=-1)

        token_lev_ans = []
        for idx, label_idx in enumerate(labels_idx):
            # ignore 'O'
            if labels_idx == 0:
                continue

            token_lev_ans += [
                    {
                        'token': self.tokenizer.convert_ids_to_tokens(int(input_ids[idx])),
                        'score': score[idx][label_idx].item(),
                        'entity': labels[label_idx]
                    }
                ]
        result.update('NER tags',token_lev_ans)
        
        results.append(result)

    return results

def main(cli_args):
    # Read from config file and make args
    with open(os.path.join(cli_args.config_dir, cli_args.task, cli_args.config_file)) as f:
        args = AttrDict(json.load(f))
    logger.info("Testing parameters {}".format(args))

    args.output_dir = os.path.join(args.ckpt_dir, args.output_dir)

    init_logger()
    set_seed(args)

    processor = processors[args.task](args)
    labels = processor.get_labels()
    config = CONFIG_CLASSES[args.model_type].from_pretrained(
        args.model_name_or_path,
        num_labels=tasks_num_labels[args.task],
        id2label={str(i): label for i, label in enumerate(labels)},
        label2id={label: i for i, label in enumerate(labels)},
    )
    tokenizer = TOKENIZER_CLASSES[args.model_type].from_pretrained(
        args.model_name_or_path,
        do_lower_case=args.do_lower_case
    )
    
    # GPU or CPU
    args.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    test_dataset = get_sentence_list(args.data_dir,args.test_file)

    checkpoints = list(
        os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + "pytorch_model.bin", recursive=True))
    )

    if not args.eval_all_checkpoints:
        checkpoints = checkpoints[-1:]

    else:
        logging.getLogger("transformers.configuration_utils").setLevel(logging.WARN)  # Reduce logging
        logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    
    logger.info("Evaluate the following checkpoints: %s", checkpoints)

    for checkpoint in checkpoints:
        global_step = checkpoint.split("-")[-1]
        model = MODEL_FOR_TOKEN_CLASSIFICATION[args.model_type].from_pretrained(checkpoint)
        model.to(args.device)
        result = test(args, model, tokenizer, lables, test_dataset, mode="test", global_step=global_step)
        results.extend(result)

    output_eval_file = os.path.join(args.test_output_dir, "ner_tagged_questions.json")
    with open(output_eval_file, "w") as f_w:
        json.dump(results,f_w,indent=4,ensure_ascii=False)

if __name__ == "__main__":
    cli_parser = argparse.ArgumentParser()

    cli_parser.add_argument("--task", type=str, required=True)
    cli_parser.add_argument("--config_dir", type=str, default='test_config')
    cli_parser.add_argument("--config_file", type=str, default="koelectra-base-v3.json")

    cli_args = cli_parser.parse_args()

    main(cli_args)
