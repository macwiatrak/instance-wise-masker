import os
from typing import Dict, Tuple

from pytorch_lightning import seed_everything
from torch.utils.data import DataLoader

from data.bc5cdr_data_reader import load_bc5cdr_datasets_to_dls
from data.mmst21pv_sample_data_reader import get_mmst21pv_sample_dataloaders
from data.utils import read_json
from models.inmasker import INMasker
from models.simple import SimpleClassifier
from runners.argparser import get_args
from runners.trainer import get_trainer
from runners.utils.utils import write_dict


def get_data(
        input_path: str,
        label_vocab_file_path: str,
        dataset: str,
        batch_size: int,
        num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    if dataset == 'mmst21pv_sample':
        train_dl, val_dl, test_dl, num_classes = get_mmst21pv_sample_dataloaders(
            input_file_path=input_path,
            label_vocab_file_path=label_vocab_file_path,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    elif dataset == 'bc5cdr':
        num_classes = len(read_json(label_vocab_file_path))
        train_dl, val_dl, test_dl = load_bc5cdr_datasets_to_dls(
            input_dir=input_path,
            batch_size=batch_size,
            num_workers=num_workers
        )
    else:
        raise ValueError(f"Dataset {dataset}  not available")
    return train_dl, val_dl, test_dl, num_classes


def main(params: Dict):
    seed_everything(params['seed_val'])
    train_dl, val_dl, test_dl, num_classes = get_data(
        input_path=params['input_data_path'],
        label_vocab_file_path=params['label_vocab_file_path'],
        batch_size=params['batch_size'],
        num_workers=params['num_workers'],
        dataset=params['dataset'],
    )

    params['num_classes'] = num_classes
    params['train_set_len'] = len(train_dl) * params['batch_size']
    write_dict(output_dict=params, output_file_path=os.path.join(params['output_dir'], 'params.json'))
    model = INMasker(params) if params['model_type'] == 'inmasker' else SimpleClassifier(params)
    trainer = get_trainer(params)

    if params['test'] and params['checkpoint_path']:
        model = model.load_from_checkpoint(params['checkpoint_path'], params)
        trainer.test(model, test_dataloaders=test_dl)
    else:
        trainer.fit(model, train_dataloader=train_dl, val_dataloaders=val_dl)

        if params['test']:
            trainer.test(ckpt_path="best", test_dataloaders=test_dl)


if __name__ == '__main__':
    args = get_args()
    main(args.__dict__)
