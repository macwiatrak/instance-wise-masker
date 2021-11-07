import os

from data.mmst21pv_sample_data_reader import get_mmst21pv_sample_dataloaders


def test_get_dataloaders():
    data_dir = 'data/tokenised-data'
    train_dl, val_dl, test_dl, num_classes = get_mmst21pv_sample_dataloaders(
        input_file_path=os.path.join(data_dir, 'inmasker_tokenised_data_mm_toy.jsonl'),
        label_vocab_file_path=os.path.join(data_dir, 'label_vocab_mm_toy.json'),
        batch_size=32,
        num_workers=4
    )
    assert len(train_dl) == 30
    assert len(val_dl) == 10
    assert len(test_dl) == 11
    assert num_classes == 10
