import pytest
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping

TEST_CONFIG_HIDDEN_DIM = 16


@pytest.fixture()
def dummy_test_data():
    input_ids = torch.tensor([[1, 5, 6, 7, 9, 0, 0, 0],
                              [1, 3, 4, 8, 3, 2, 9, 0],
                              [1, 2, 6, 4, 2, 2, 8, 9],
                              [1, 3, 9, 0, 0, 0, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0],
                                   [1, 1, 1, 1, 1, 1, 1, 0],
                                   [1, 1, 1, 1, 1, 1, 1, 1],
                                   [1, 1, 1, 0, 0, 0, 0, 0]])
    entity_mask = torch.tensor([[0, 0, 1, 0, 0, 0, 0, 0],
                                [0, 0, 0, 0, 1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0, 1, 0],
                                [0, 1, 0, 0, 0, 0, 0, 0]])
    labels = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0], [1, 0, 0]])
    return input_ids, attention_mask, entity_mask, labels


@pytest.fixture()
def test_hparams():
    return {
        'classifier_dropout': 0.2,
        'max_seq_len': 8,
        'pretrained_bert_path': None,
        'encoder_config_file_path': 'test_model_config.json',
        'lambda': 0.1,
        'mask_token_id': 11,
        'eps': 1e-8,
        'num_classes': 3,
        'lr': 0.01,
        'max_epochs': 100,
        'patience': 10,
        'monitor_quantity': 'val_total_loss',
        'token_importance_method': 'basic',
        'seed_val': 42,
        'warmup_proportion': 0.,
        'full_multi_task_learning': False,
        'critic_multi_task_learning': False,
        'train_set_len': 4,
        'batch_size': 2,
        'simple_classifier_l2_penalty': False,
    }


@pytest.fixture()
def test_trainer_inmasker():
    early_stop_callback = EarlyStopping(monitor='val_total_loss',
                                        patience=20,
                                        mode='min')
    return Trainer(
        gpus=None,
        logger=False,
        max_epochs=40,
        gradient_clip_val=0.,
        progress_bar_refresh_rate=1,
        callbacks=[early_stop_callback],
        checkpoint_callback=False,
    )


@pytest.fixture()
def test_trainer_simple_classifier():
    early_stop_callback = EarlyStopping(monitor='val_loss',
                                        patience=20,
                                        mode='min')
    return Trainer(
        gpus=None,
        logger=False,
        max_epochs=100,
        gradient_clip_val=0.,
        progress_bar_refresh_rate=1,
        callbacks=[early_stop_callback],
        checkpoint_callback=False,
    )
