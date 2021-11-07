import pytest
import torch

from models.inmasker import INMasker


@pytest.mark.parametrize(
    ['full_mtl', 'critic_mtl', 'token_importance_method'],
    [(False, False, 'basic'), (False, False, 'attention'),
     (True, False, 'basic'), (True, False, 'attention'),
     (False, True, 'basic'), (False, True, 'attention')])
def test_inmasker_forward_pass(test_hparams, dummy_test_data, full_mtl, critic_mtl, token_importance_method):
    test_hparams['full_multi_task_learning'] = full_mtl
    test_hparams['critic_multi_task_learning'] = critic_mtl
    test_hparams['token_importance_method'] = token_importance_method
    model = INMasker(test_hparams)
    total_loss = model.training_step(dummy_test_data, batch_nb=0)
    assert total_loss != 0

    output_dict = model.validation_step(dummy_test_data, batch_nb=0)
    assert output_dict['total_loss'] != 0
    assert output_dict['critic_one_loss'] != 0
    assert output_dict['actor_loss'] != 0
    assert output_dict['critic_two_loss'] != 0
    assert output_dict['critic_one_out'].shape == torch.Size([4, 3])
    assert output_dict['critic_two_out'].shape == torch.Size([4, 3])
    assert output_dict['actor_out'].shape == torch.Size([4, 3])
    assert output_dict['labels'].shape == torch.Size([4])
