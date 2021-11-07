from typing import Dict, List, Tuple

import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from data.utils import read_jsonl, read_json


def get_mmst21pv_sample_dataset(data: List[Dict], label_vocab: Dict, num_classes: int) -> TensorDataset:

    input_ids = torch.stack([torch.tensor(item['token_ids'], dtype=torch.long) for item in data])
    attention_mask = torch.stack([torch.tensor(item['attention_mask'], dtype=torch.long) for item in data])
    entity_mask = torch.stack([torch.tensor(item['entity_mask'], dtype=torch.float32) for item in data])
    labels = torch.stack(
        [F.one_hot(torch.tensor(label_vocab[str(item['labels'])]), num_classes=num_classes) for item in data])
    return TensorDataset(
        input_ids,
        attention_mask,
        entity_mask,
        labels
    )


def get_mmst21pv_sample_dataloaders(
        input_file_path: str,
        label_vocab_file_path: str,
        batch_size: int,
        num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:

    train_samples = [item for item in read_jsonl(input_file_path) if item['split'] == 'train']
    val_samples = [item for item in read_jsonl(input_file_path) if item['split'] == 'val']
    test_samples = [item for item in read_jsonl(input_file_path) if item['split'] == 'test']

    label_vocab = read_json(label_vocab_file_path)
    num_classes = len(label_vocab)

    train_dataset = get_mmst21pv_sample_dataset(train_samples, label_vocab, num_classes)
    val_dataset = get_mmst21pv_sample_dataset(val_samples, label_vocab, num_classes)
    test_dataset = get_mmst21pv_sample_dataset(test_samples, label_vocab, num_classes)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers)

    return train_dataloader, val_dataloader, test_dataloader, num_classes
