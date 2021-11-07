import os
from typing import List, Tuple

import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer

from data.preprocess_bc5cdr import preprocess_bc5cdr
from data.utils import Bc5cdrSample, read_jsonl, read_json

TRAIN_DATASET = 'train_dataset.pth'
VAL_DATASET = 'val_dataset.pth'
TEST_DATASET = 'test_dataset.pth'


def get_bc5cdr_dataset(data: List[Bc5cdrSample]) -> TensorDataset:
    return TensorDataset(
        torch.stack([sample.input_ids for sample in data]),
        torch.stack([sample.attention_mask for sample in data]),
        torch.stack([sample.entity_mask for sample in data]),
        torch.stack([sample.label for sample in data]),
    )


def get_bc5cdr_dataloaders(
        input_file_path: str,
        label_vocab_file_path: str,
        train_test_split_file_path: str,
        tokenizer_path: str,
        max_seq_len: int,
        batch_size: int,
        num_workers: int = 8
):
    docs = read_jsonl(input_file_path)
    label_vocab = read_json(label_vocab_file_path)
    split = read_json(train_test_split_file_path)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    train_samples, val_samples, test_samples = preprocess_bc5cdr(
        tokenizer=tokenizer,
        label_vocab=label_vocab,
        docs=docs,
        max_seq_len=max_seq_len,
        split=split,
    )

    train_dl = DataLoader(
        get_bc5cdr_dataset(train_samples), shuffle=True, batch_size=batch_size, num_workers=num_workers)
    val_dl = DataLoader(
        get_bc5cdr_dataset(val_samples), shuffle=False, batch_size=batch_size, num_workers=num_workers)
    test_dl = DataLoader(
        get_bc5cdr_dataset(test_samples), shuffle=False, batch_size=batch_size, num_workers=num_workers)

    return train_dl, val_dl, test_dl


def save_datasets(
        input_file_path: str,
        label_vocab_file_path: str,
        train_test_split_file_path: str,
        tokenizer_path: str,
        max_seq_len: int,
        output_dir: str,
):
    docs = read_jsonl(input_file_path)
    label_vocab = read_json(label_vocab_file_path)
    split = read_json(train_test_split_file_path)
    tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

    train_samples, val_samples, test_samples = preprocess_bc5cdr(
        tokenizer=tokenizer,
        label_vocab=label_vocab,
        docs=docs,
        max_seq_len=max_seq_len,
        split=split,
    )

    torch.save(get_bc5cdr_dataset(train_samples), os.path.join(output_dir, TRAIN_DATASET))
    torch.save(get_bc5cdr_dataset(val_samples), os.path.join(output_dir, VAL_DATASET))
    torch.save(get_bc5cdr_dataset(test_samples), os.path.join(output_dir, TEST_DATASET))


def load_bc5cdr_datasets_to_dls(
        input_dir: str,
        batch_size: int,
        num_workers: int = 8,
) -> Tuple[DataLoader, DataLoader, DataLoader]:

    with wrapped_s3_directory_reader(input_dir) as local_dir:
        train_dataset = torch.load(os.path.join(local_dir, TRAIN_DATASET))
        val_dataset = torch.load(os.path.join(local_dir, VAL_DATASET))
        test_dataset = torch.load(os.path.join(local_dir, TEST_DATASET))

    train_dl = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    val_dl = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    test_dl = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers)

    return train_dl, val_dl, test_dl


if __name__ == '__main__':
    pass
    # save_datasets(
    #     input_file_path='/Users/maciejwiatrak/biocreative-v-all.json',
    #     label_vocab_file_path='/Users/maciejwiatrak/instance-wise-masker/data/processed-data/label_vocab_bc5cdr.json',
    #     train_test_split_file_path='/Users/maciejwiatrak/biocreative_bc5cdr_train_val_test_split.json',
    #     tokenizer_path='allenai/scibert_scivocab_uncased',
    #     max_seq_len=128,
    #     output_dir='/tmp/model/'
    # )
    # train_dl, val_dl, test_dl = load_bc5cdr_datasets_to_dls(
    #     input_dir='/tmp/model/',
    #     batch_size=32,
    #     num_workers=8,
    # )
    # assert 1
