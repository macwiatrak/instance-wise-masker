import torch

from models.encoders import TextEncoder
from tests.conftest import TEST_CONFIG_HIDDEN_DIM


def test_text_encoder(test_hparams, dummy_test_data):
    model = TextEncoder(test_hparams, model='critic_one')
    input_ids, attention_mask, entity_mask, _ = dummy_test_data

    out_w_entity_mask = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        entity_mask=entity_mask,
    )
    assert out_w_entity_mask[0].shape == torch.Size(
        [input_ids.shape[0], TEST_CONFIG_HIDDEN_DIM])

    out_wo_entity_mask = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    assert out_wo_entity_mask[1].shape == torch.Size(
        [input_ids.shape[0], input_ids.shape[1], TEST_CONFIG_HIDDEN_DIM])
