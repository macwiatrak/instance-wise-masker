import pytest
from torch.utils.data import TensorDataset, DataLoader

from models.inmasker import INMasker
from models.simple import SimpleClassifier


@pytest.mark.parametrize(
    ['full_mtl', 'critic_mtl', 'token_importance_method'],
    [(False, False, 'basic'), (False, False, 'attention'),
     (True, False, 'basic'), (True, False, 'attention'),
     (False, True, 'basic'), (False, True, 'attention')])
def test_train_inmasker_with_dummy_data(test_hparams, dummy_test_data, test_trainer_inmasker,
                                        full_mtl, critic_mtl, token_importance_method):
    test_hparams['full_multi_task_learning'] = full_mtl
    test_hparams['critic_multi_task_learning'] = critic_mtl
    test_hparams['token_importance_method'] = token_importance_method
    # this test only tests whether the model trains and runs
    input_ids, attention_mask, entity_mask, one_hot_labels = dummy_test_data
    dataset = TensorDataset(input_ids, attention_mask, entity_mask, one_hot_labels)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    model = INMasker(test_hparams)
    test_trainer_inmasker.fit(model, train_dataloader=dataloader, val_dataloaders=dataloader)
    assert isinstance(model, INMasker)  # dummy assertion


def test_train_simple_model_with_dummy_data(test_hparams, dummy_test_data, test_trainer_simple_classifier):
    # this test only tests whether the model trains and runs
    input_ids, attention_mask, entity_mask, one_hot_labels = dummy_test_data
    dataset = TensorDataset(input_ids, attention_mask, entity_mask, one_hot_labels)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    model = SimpleClassifier(test_hparams)
    test_trainer_simple_classifier.fit(model, train_dataloader=dataloader, val_dataloaders=dataloader)
    assert isinstance(model, SimpleClassifier)  # dummy assertion
