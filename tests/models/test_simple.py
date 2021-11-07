import pytest
import torch

from models.simple import SimpleClassifier


@pytest.mark.parametrize('l2_norm_penalty', [True, False])
def test_simple_classifier_forward_pass(test_hparams, dummy_test_data, l2_norm_penalty):
    test_hparams['simple_classifier_l2_penalty'] = l2_norm_penalty
    model = SimpleClassifier(test_hparams)
    total_loss = model.training_step(dummy_test_data, batch_nb=0)
    assert total_loss != 0

    output_dict = model.validation_step(dummy_test_data, batch_nb=0)
    assert output_dict['loss'] != 0
    assert output_dict['logits'].shape == torch.Size([4, 3])
    assert output_dict['labels'].shape == torch.Size([4])
