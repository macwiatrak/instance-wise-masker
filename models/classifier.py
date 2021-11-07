from typing import Dict, Tuple

import torch
from torch import nn

from models.encoders import TextEncoder


class TextClassifier(nn.Module):
    def __init__(self, hparams: Dict, model: str):
        super().__init__()

        self.params = hparams
        self.model = model

        self.encoder = TextEncoder(hparams=hparams, model=model)
        self.decoder = nn.Linear(self.encoder.output_dim, hparams['num_classes'])
        self.dropout = nn.Dropout(hparams['classifier_dropout'])
        self.sigmoid = nn.Sigmoid()
        self.eps = hparams['eps']

        if model == 'actor' and hparams['token_importance_method'] == 'basic':
            self.token_importance_scorer = nn.Linear(self.encoder.output_dim, 1)
            if hparams['token_importance_method'] == 'basic_with_frozen_layer':
                self.token_importance_scorer.weight.requires_grad = False
                self.token_importance_scorer.weight.requires_grad = False

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            entity_mask: torch.Tensor,
            return_token_importance: bool = False,
    ):
        entity_repr, token_reprs, attention_out = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            entity_mask=entity_mask
        )
        decoder_out = self.decoder(self.dropout(entity_repr))
        if not return_token_importance:
            return decoder_out
        token_importance = self.get_token_importance(
            token_reprs=token_reprs,
            attention_mask=attention_mask,
            attention_out=attention_out,
            entity_mask=entity_mask
        )
        return decoder_out, token_importance

    def get_token_importance(
            self,
            token_reprs: torch.Tensor,
            attention_mask: torch.Tensor,
            entity_mask: torch.Tensor,
            attention_out: Tuple,
    ):
        if self.params['token_importance_method'] == 'basic' or \
                self.params['token_importance_method'] == 'basic_with_frozen_layer':
            return self.basic_token_importance(token_reprs, attention_mask)
        elif self.params['token_importance_method'] == 'attention':
            return self.attention_token_importance(
                attention_output=attention_out,
                entity_mask=entity_mask,
                attention_mask=attention_mask
            )
        else:
            raise ValueError(f"{self.params['token_importance_method']} not available.")

    def basic_token_importance(
            self,
            token_reprs: torch.Tensor,
            attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        decoder_out = self.token_importance_scorer(self.dropout(token_reprs))
        probs = self.sigmoid(decoder_out.squeeze(-1))
        probs = torch.where(attention_mask.bool(), probs, torch.zeros_like(probs))
        return probs

    def attention_token_importance(
            self,
            attention_output: Tuple,
            entity_mask: torch.tensor,
            attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # iterate through layers
        mean_attention = torch.cat([atts.mean(dim=1).unsqueeze(1) for atts in attention_output], dim=1).mean(dim=1)

        # create entity_attention_mask for entity tokens we later use for predictions
        batch_size = entity_mask.shape[0]
        seq_len = entity_mask.shape[1]
        # make [batch_size x seq_len] into [batch_size x seq_len x seq_len]
        # pick attention values for entity tokens
        entity_attention_mask = torch.transpose(
            entity_mask.repeat(1, seq_len).view(batch_size, seq_len, seq_len), 1, 2)
        # sum attention for all entity tokens and
        nr_entity_tokens = torch.count_nonzero(entity_mask.clone().detach(), dim=1).float().unsqueeze(-1)
        att_token_probs = (entity_attention_mask * mean_attention).sum(dim=1).div(nr_entity_tokens)

        nr_tokens = torch.count_nonzero(attention_mask.clone().detach(), dim=1).float().unsqueeze(-1)
        inv_softmax_att = torch.log(att_token_probs + self.eps) + torch.log(nr_tokens)
        att_token_importance = torch.where(
            attention_mask.bool(), self.sigmoid(inv_softmax_att), torch.zeros_like(inv_softmax_att))
        return att_token_importance
