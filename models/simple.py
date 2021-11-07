from typing import Dict, Tuple, List, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics.functional import accuracy
from transformers import get_linear_schedule_with_warmup

from models.classifier import TextClassifier


class SimpleClassifier(pl.LightningModule):
    def __init__(self, hparams: Dict):
        super(SimpleClassifier, self).__init__()

        self.params = hparams

        self.model = TextClassifier(hparams=hparams, model='actor')
        self.num_classes = hparams['num_classes']
        self.lamda = hparams['lambda']

    def step(self, batch, mode: str) -> Tuple[torch.Tensor, torch.Tensor]:

        input_ids, attention_mask, entity_mask, one_hot_labels = batch
        labels = torch.argmax(one_hot_labels, dim=1).long()

        logits, token_importance = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            entity_mask=entity_mask,
            return_token_importance=True,
        )

        if self.params['simple_classifier_l2_penalty']:
            l2_penalty = torch.norm(token_importance, p=2, dim=-1).sum().div(token_importance.shape[0])
            loss = F.cross_entropy(logits, labels) + self.lamda * l2_penalty
        else:
            loss = F.cross_entropy(logits, labels)

        # log loss
        self.log(f'{mode}_loss', loss, prog_bar=True, logger=True)
        return loss, logits

    def training_step(self, batch, batch_nb):
        loss, logits = self.step(batch, mode='train')
        return loss

    def validation_step(self, batch, batch_nb):
        _, _, _, one_hot_labels = batch
        loss, logits = self.step(batch, mode='val')
        return {'loss': loss, 'logits': logits, 'labels': torch.argmax(one_hot_labels, dim=1).long()}

    def test_step(self, batch, batch_nb):
        _, _, _, one_hot_labels = batch
        loss, logits = self.step(batch, mode='test')
        return {'loss': loss, 'logits': logits, 'labels': torch.argmax(one_hot_labels, dim=1).long()}

    def eval_epoch_end(self, outputs: List[Dict], mode: str) -> None:
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        logits = torch.cat([x['logits'] for x in outputs], dim=0)
        labels = torch.cat([x['labels'] for x in outputs], dim=0)

        micro_acc = accuracy(preds=logits, target=labels, average='micro')
        macro_acc = accuracy(preds=logits, target=labels, average='macro',
                             num_classes=self.num_classes)

        self.log_dict({
            f'{mode}_loss': avg_loss,
            f'{mode}_micro_acc': micro_acc,
            f'{mode}_macro_acc': macro_acc,
        }, prog_bar=True, logger=True)

    def validation_epoch_end(self, outputs: List[Dict]) -> None:
        self.eval_epoch_end(outputs, mode='val')

    def test_epoch_end(self, outputs: List[Dict]) -> None:
        self.eval_epoch_end(outputs, mode='test')

    def configure_optimizers(self):
        optim = torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=self.params['lr'])
        if self.params['warmup_proportion'] == 0.:
            return optim
        scheduler = self.get_scheduler(optim)
        return [optim], [{
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
            'reduce_on_plateau': False,
            'monitor': 'val_loss'
        }]

    def get_scheduler(self, optimiser: Optional[torch.optim.Adam]):
        num_train_steps = int(self.params['train_set_len'] / self.params['batch_size']) * self.params['max_epochs']
        num_warmup_steps = int(num_train_steps * self.params['warmup_proportion'])
        return get_linear_schedule_with_warmup(optimizer=optimiser, num_training_steps=num_train_steps,
                                               num_warmup_steps=num_warmup_steps)
