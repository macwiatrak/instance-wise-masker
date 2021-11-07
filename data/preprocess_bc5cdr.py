from typing import Dict, Generator, Tuple, List

import torch
from torch.nn import functional as F
from tqdm import tqdm
from transformers import BertTokenizer

from data.utils import Bc5cdrSample


def process_sample(
        tokenizer: BertTokenizer,
        doc_text: str,
        start: int,
        end: int,
        max_seq_len: int,
        label: int,
        num_classes: int,
) -> Bc5cdrSample:
    left_context_tokens = tokenizer.tokenize(doc_text[start:])
    right_context_tokens = tokenizer.tokenize(doc_text[end:])
    mention_ids = tokenizer.tokenize(doc_text[start:end])

    left_quota = (max_seq_len - len(mention_ids)) // 2 - 1
    right_quota = max_seq_len - len(mention_ids) - left_quota - 2
    left_add = len(left_context_tokens)
    right_add = len(right_context_tokens)
    if left_add <= left_quota:
        if right_add > right_quota:
            right_quota += left_quota - left_add
    else:
        if right_add <= right_quota:
            left_quota += right_quota - right_add

    context_tokens = (
            ["[CLS]"] + left_context_tokens[-left_quota:] + mention_ids + right_context_tokens[:right_quota] + ["[SEP]"]
    )
    entity_mask = (
            [0.] * (len(left_context_tokens[-left_quota:]) + 1) + [1.] * len(mention_ids) +
            [0.] * (len(right_context_tokens[:right_quota]) + 1)
    )
    input_ids = tokenizer.convert_tokens_to_ids(context_tokens)
    padding = [0] * (max_seq_len - len(input_ids))
    attention_mask = [1] * len(input_ids) + padding
    input_ids += padding
    entity_mask += padding
    assert len(input_ids) == len(attention_mask) == len(entity_mask) == max_seq_len
    return Bc5cdrSample(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        attention_mask=torch.tensor(attention_mask, dtype=torch.long),
        entity_mask=torch.tensor(entity_mask, dtype=torch.float32),
        label=F.one_hot(torch.tensor(label), num_classes=num_classes),
    )


def preprocess_bc5cdr(
        tokenizer: BertTokenizer,
        label_vocab: Dict,
        docs: Generator,
        max_seq_len: int,
        split: Dict
) -> Tuple[List[Bc5cdrSample], List[Bc5cdrSample], List[Bc5cdrSample]]:
    train = []
    val = []
    test = []
    for doc in tqdm(docs):
        full_text = doc["title"] + " " + doc['abstract']
        for annot in doc['annotations']:
            sample = process_sample(
                tokenizer=tokenizer,
                doc_text=full_text,
                start=annot['start'],
                end=annot['end'],
                max_seq_len=max_seq_len,
                label=label_vocab[annot['identifier']],
                num_classes=len(label_vocab)
            )
            if doc['document_id'] in split['train']:
                train.append(sample)
            elif doc['document_id'] in split['val']:
                val.append(sample)
            else:
                test.append(sample)
    return train, val, test
