import json
from typing import Generator, Dict, NamedTuple

import torch


def read_jsonl(file_path: str) -> Generator:
    with open(file_path, 'r') as f:
        for line in f:
            yield json.loads(line)


def read_json(file_path: str) -> Dict:
    with open(file_path, 'r') as f:
        return json.load(f)


class Bc5cdrSample(NamedTuple):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    entity_mask: torch.Tensor
    label: torch.Tensor
