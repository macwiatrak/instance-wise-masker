# Codebase for INMasker, "Instance-Wise Text Masking for Model Regularization"

This repository contains implementation of the INMasker model.

## Requirements & setup

The model uses `python 3.8`. Full list of requirements can be found in `requirements.txt`. In order to setup the repo please execute from the root dir of the repo:

```python
pip install -r requirements.txt
```
followed by:
```python
pip install -e .
```
We also include `Dockerfile` in the repo.

### Data

We provide two datasets used for running experiments:
- MedMentions toy:
    - Path to the tokenised data - `data/processed-data/inmasker_tokenised_data_mm_toy.jsonl`
    - Path to the label vocab - `data/processed-data/label_vocab_mm_toy.json`
- BC5CDR:
    - Path to the Google Drive folder with the tokenised data - https://drive.google.com/drive/folders/1xIz-KOCDGQlIF5MLseDPlypEY2OV6DdC?usp=sharing
    - To run the experiments on the BC5CDR data, download the Google Drive folder linked above, in case of any problems please reach out to `macwiatrak@gmail.com`

### Example commands

Below, we provide some example commands for the two INMasker models (basic & attention) and two baseline Classifier models (simple and with `l2` penalty) on the `MedMentions Toy` dataset.

Classifier:
```python
python runners/train.py --model_type simple_classifier --monitor_quantity val_micro_acc --lr 0.001 \
    --dataset mmst21pv_sample --input_data_path data/processed-data/inmasker_tokenised_data_mm_toy.jsonl \
    --label_vocab_file_path data/processed-data/label_vocab_mm_toy.json \
    --encoder_config_file_path runners/utils/small_transformer_config.json --test
```

Classifier with `l2` penalty:
```python
python runners/train.py --model_type simple_classifier --monitor_quantity val_micro_acc --lr 0.001 \
    --dataset mmst21pv_sample --input_data_path data/processed-data/inmasker_tokenised_data_mm_toy.jsonl \
    --label_vocab_file_path data/processed-data/label_vocab_mm_toy.json \
    --encoder_config_file_path runners/utils/small_transformer_config.json --test \
    --simple_classifier_l2_penalty --lambda 0.1
```

INMasker basic:
```python
python runners/train.py --model_type inmasker --monitor_quantity val_actor_micro_acc --lr 0.001 \
    --dataset mmst21pv_sample --input_data_path data/processed-data/inmasker_tokenised_data_mm_toy.jsonl \
    --label_vocab_file_path data/processed-data/label_vocab_mm_toy.json \
    --encoder_config_file_path runners/utils/small_transformer_config.json --test \
    --token_importance_method basic --critic_multi_task_learning
```

INMasker attention:
```python
python runners/train.py --model_type inmasker --monitor_quantity val_actor_micro_acc --lr 0.001 \
    --dataset mmst21pv_sample --input_data_path data/processed-data/inmasker_tokenised_data_mm_toy.jsonl \
    --label_vocab_file_path data/processed-data/label_vocab_mm_toy.json \
    --encoder_config_file_path runners/utils/small_transformer_config.json --test \
    --token_importance_method attention --critic_multi_task_learning
```

Following commands can also be used to run the models on the BC5CDR dataset, however, following arguments should be changes:
- `--dataset bc5cdr`
- `--input_data_path <input your local directory path to bc5cdr after downloading it>`
- `--label_vocab_file_path <input your local file path to bc5cdr after downloading it>`
- `--encoder_config_file_path runners/utils/large_transformer_config.json`
- `--lr 0.0005`

For full list of available command inputs, please see `runners/argparser.py` file.

### Model configs

We use the Transformer architecture from Hugging Face implementation as our encoders. We provide two configs used for conducting the experiments inside the repo:
- Small (repo path: `runners/utils/small_transformer_config.json`)
- Large (repo path: `runners/utils/large_transformer_config.json`)


### Outputs
The model outputs:

- Test Micro accuracy
- Test Macro accuracy
- Test Loss

It's also possible to monitor the training and validation progress looking at the progress bar.

### Tests

We also include some unit tests for testing and debugging purposes. We use `pytest`.
