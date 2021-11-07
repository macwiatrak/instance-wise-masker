import json
from typing import Dict


def write_dict(output_dict: Dict, output_file_path: str):
    with open(output_file_path, 'w') as f:
        json.dump(output_dict, f)
