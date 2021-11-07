from typing import Dict

import torch
from torch import nn
from transformers import AutoModel, AutoConfig, BertModel

BERT_HIDDEN_SIZE = 768
SMALL_TRANSFORMER_HIDDEN_SIZE = 16
LARGE_TRANSFORMER_HIDDEN_SIZE = 128


class TextEncoder(nn.Module):
    def __init__(self, hparams: Dict, model: str):
        super().__init__()

        self.params = hparams
        self.model = model

        if hparams["pretrained_bert_path"] is not None:
            self.encoder = BertModel.from_pretrained(hparams["pretrained_bert_path"], output_attentions=True)
            self.output_dim = BERT_HIDDEN_SIZE
        else:
            self.encoder = AutoModel.from_config(AutoConfig.from_pretrained(hparams['encoder_config_file_path']))
            # TODO: pass transformer config through argparser
            self.output_dim = SMALL_TRANSFORMER_HIDDEN_SIZE if 'small' in hparams['encoder_config_file_path'] else LARGE_TRANSFORMER_HIDDEN_SIZE  # noqa

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            entity_mask: torch.Tensor = None,
    ):
        entity_repr = None
        encoder_out = self.encoder(input_ids, attention_mask=attention_mask)
        if entity_mask is not None:
            # Average representation of MASK (i.e. an entity)
            entity_repr = (torch.einsum('ij, ijk -> ik', entity_mask.float(), encoder_out.last_hidden_state) /
                           torch.unsqueeze(torch.einsum('ij -> i', entity_mask.float()), -1))  # so shape is i1 for broadcast  # noqa
        return entity_repr, encoder_out.last_hidden_state, encoder_out.attentions
