import pytest
import torch

from models.classifier import TextClassifier


def test_classifier_critic(test_hparams, dummy_test_data):
    model = TextClassifier(test_hparams, model='critic_one')
    input_ids, attention_mask, entity_mask, _ = dummy_test_data
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        entity_mask=entity_mask
    )
    assert out.shape == torch.Size([input_ids.shape[0], test_hparams['num_classes']])


@pytest.mark.parametrize('token_importance_method', ['basic', 'attention'])
def test_classifier_actor(test_hparams, dummy_test_data, token_importance_method):
    test_hparams['token_importance_method'] = token_importance_method
    model = TextClassifier(test_hparams, model='actor')
    input_ids, attention_mask, entity_mask, _ = dummy_test_data
    dec_out, token_importance = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        entity_mask=entity_mask,
        return_token_importance=True,
    )
    assert dec_out.shape == torch.Size([input_ids.shape[0], test_hparams['num_classes']])
    assert token_importance.shape == input_ids.shape
