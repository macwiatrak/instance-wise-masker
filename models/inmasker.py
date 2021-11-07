import random
from typing import Dict, Tuple, List, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics.functional import accuracy
from transformers import get_linear_schedule_with_warmup

from models.classifier import TextClassifier


class INMasker(pl.LightningModule):
    def __init__(self, hparams: Dict):
        super(INMasker, self).__init__()

        self.params = hparams

        self.actor = TextClassifier(hparams=hparams, model='actor')
        if hparams['full_multi_task_learning']:
            self.critic_one = self.actor
            self.critic_two = self.actor
        elif hparams['critic_multi_task_learning']:
            self.critic_one = TextClassifier(hparams=hparams, model='critic_one')
            self.critic_two = self.critic_one
        else:
            self.critic_one = TextClassifier(hparams=hparams, model='critic_one')
            self.critic_two = TextClassifier(hparams=hparams, model='critic_two')

        self.softmax = torch.nn.Softmax(dim=1)

        self.lamda = hparams['lambda']
        self.mask_token_id = hparams['mask_token_id']
        self.eps = hparams['eps']
        self.num_classes = hparams['num_classes']

    def step(self, batch, mode: str) -> Tuple[
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        input_ids, attention_mask, entity_mask, one_hot_labels = batch
        labels = torch.argmax(one_hot_labels, dim=1).long()

        # calculate token importance
        actor_out, token_importance = self.actor(
            input_ids=input_ids,
            attention_mask=attention_mask,
            entity_mask=entity_mask,
            return_token_importance=True,
        )
        token_selection = torch.bernoulli(token_importance).detach()
        critic_one_input_ids, critic_two_input_ids = self.get_critic_input_ids(
            input_ids=input_ids,
            token_selection=token_selection,
            attention_mask=attention_mask,
        )

        # critic one objective
        critic_one_out = self.critic_one(
            input_ids=critic_one_input_ids,
            attention_mask=attention_mask,
            entity_mask=entity_mask,
        )
        critic_one_loss = F.cross_entropy(critic_one_out, labels)

        # critic two objective
        critic_two_out = self.critic_two(
            input_ids=critic_two_input_ids,
            attention_mask=attention_mask,
            entity_mask=entity_mask,
        )
        critic_two_loss = F.cross_entropy(critic_two_out, labels)

        actor_loss = self.actor_loss(
            actor_out=actor_out,
            critic_one_out=self.softmax(critic_one_out).clone().detach(),
            critic_two_out=self.softmax(critic_two_out).clone().detach(),
            token_selection=token_selection.clone().detach(),
            token_importance=token_importance,
            one_hot_labels=one_hot_labels.float()
        )
        total_loss = actor_loss + critic_one_loss + critic_two_loss

        # log stats
        self.log_dict({
            f'{mode}_total_loss': total_loss,
            f'{mode}_actor_loss': actor_loss,
            f'{mode}_critic_one_loss': critic_one_loss,
            f'{mode}_critic_two_loss': critic_two_loss
        }, prog_bar=True, logger=True)
        return actor_loss, critic_one_loss, critic_two_loss, actor_out, critic_one_out, critic_two_out

    def training_step(self, batch, batch_nb):
        s_loss, p_loss, b_loss, _, _, _ = self.step(batch, mode='train')
        total_loss = s_loss + p_loss + b_loss
        return total_loss

    def validation_step(self, batch, batch_nb):
        _, _, _, one_hot_labels = batch
        actor_loss, c1_loss, c2_loss, actor_out, c1_out, c2_out = self.step(batch, mode='val')
        total_loss = actor_loss + c1_loss + c2_loss
        # pass through loss and predictions to later compute micro and macro accuracy
        return {
            'total_loss': total_loss,
            'actor_loss': actor_loss,
            'critic_one_loss': c1_loss,
            'critic_two_loss': c2_loss,
            'actor_out': actor_out,
            'critic_one_out': c1_out,
            'critic_two_out': c2_out,
            'labels': torch.argmax(one_hot_labels, dim=1).long()
        }

    def test_step(self, batch, batch_nb):
        _, _, _, one_hot_labels = batch
        actor_loss, c1_loss, c2_loss, actor_out, c1_out, c2_out = self.step(batch, mode='test')
        total_loss = actor_loss + c1_loss + c2_loss
        # pass through loss and predictions to later compute micro and macro accuracy
        return {
            'total_loss': total_loss,
            'actor_loss': actor_loss,
            'critic_one_loss': c1_loss,
            'critic_two_loss': c2_loss,
            'actor_out': actor_out,
            'critic_one_out': c1_out,
            'critic_two_out': c2_out,
            'labels': torch.argmax(one_hot_labels, dim=1).long()
        }

    def eval_epoch_end(self, outputs: List[Dict], mode: str) -> None:
        avg_total_loss = torch.stack([x['total_loss'] for x in outputs]).mean()
        avg_actor_loss = torch.stack([x['actor_loss'] for x in outputs]).mean()
        avg_c1_loss = torch.stack([x['critic_one_loss'] for x in outputs]).mean()
        avg_c2_loss = torch.stack([x['critic_two_loss'] for x in outputs]).mean()
        actor_out = torch.cat([x['actor_out'] for x in outputs], dim=0)
        c1_out = torch.cat([x['critic_one_out'] for x in outputs], dim=0)
        c2_out = torch.cat([x['critic_two_out'] for x in outputs], dim=0)
        labels = torch.cat([x['labels'] for x in outputs], dim=0)

        actor_micro_acc = accuracy(preds=actor_out, target=labels, average='micro')
        actor_macro_acc = accuracy(preds=actor_out, target=labels, average='macro',
                                   num_classes=self.num_classes)

        c1_micro_acc = accuracy(preds=c1_out, target=labels, average='micro')
        c1_macro_acc = accuracy(preds=c1_out, target=labels, average='macro',
                                num_classes=self.num_classes)

        c2_micro_acc = accuracy(preds=c2_out, target=labels, average='micro')
        c2_macro_acc = accuracy(preds=c2_out, target=labels, average='macro',
                                num_classes=self.num_classes)

        self.log_dict({
            f'{mode}_total_loss': avg_total_loss,
            f'{mode}_actor_loss': avg_actor_loss,
            f'{mode}_critic_one_loss': avg_c1_loss,
            f'{mode}_critic_two_loss': avg_c2_loss,
            f'{mode}_actor_micro_acc': actor_micro_acc,
            f'{mode}_actor_macro_acc': actor_macro_acc,
            f'{mode}_critic_one_micro_acc': c1_micro_acc,
            f'{mode}_critic_one_macro_acc': c1_macro_acc,
            f'{mode}_critic_two_micro_acc': c2_micro_acc,
            f'{mode}_critic_two_macro_acc': c2_macro_acc
        }, prog_bar=True, logger=True)

    def validation_epoch_end(self, outputs: List[Dict]) -> None:
        self.eval_epoch_end(outputs, mode='val')

    def test_epoch_end(self, outputs: List[Dict]) -> None:
        self.eval_epoch_end(outputs, mode='test')

    def actor_loss(
            self,
            actor_out: torch.Tensor,
            critic_one_out: torch.Tensor,
            critic_two_out: torch.Tensor,
            token_selection: torch.Tensor,
            token_importance: torch.Tensor,
            one_hot_labels: torch.Tensor,
    ):
        actor_ce_loss = F.cross_entropy(actor_out, torch.argmax(one_hot_labels, dim=1), reduce=False)

        critic_one = -torch.sum(one_hot_labels * torch.log(critic_one_out + self.eps), dim=1)
        critic_two = -torch.sum(one_hot_labels * torch.log(critic_two_out + self.eps), dim=1)

        # policy gradient loss computation
        reg_loss = torch.abs(critic_one - critic_two) * -torch.sum(
            token_selection * torch.log(token_importance + self.eps) + (1 - token_selection) * torch.log(
                1 - token_importance + self.eps), dim=1)

        # account for the fact that different sentences have different length
        # seq_lens = torch.count_nonzero(token_importance.clone().detach(), dim=1)
        # actor_loss += self.lamda * torch.sum(token_importance, dim=1) / seq_lens
        actor_loss = actor_ce_loss + self.lamda * reg_loss
        return torch.mean(actor_loss)

    def get_critic_input_ids(
            self,
            token_selection: torch.Tensor,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # pick tokens for two critics making sure all pad tokens have selection equal to one
        critic_one_sel = torch.where(attention_mask.bool(), token_selection, torch.ones_like(token_selection))
        critic_two_sel = torch.where(attention_mask.bool(), (1 - token_selection), torch.ones_like(token_selection))

        critic_one_input_ids = torch.where(
            critic_one_sel.bool(), input_ids, torch.full_like(input_ids, self.mask_token_id))
        critic_two_input_ids = torch.where(
            critic_two_sel.bool(), input_ids, torch.full_like(input_ids, self.mask_token_id))

        # randomly shuffle two critic inputs
        critic_inputs = [critic_one_input_ids, critic_two_input_ids]
        random.shuffle(critic_inputs)

        return critic_inputs[0], critic_inputs[1]

    def configure_optimizers(self):
        optim = torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=self.params['lr'])
        if self.params['warmup_proportion'] > 0.:
            return optim
        scheduler = self.get_scheduler(optim)
        return [optim], [{
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1,
            'reduce_on_plateau': False,
            'monitor': 'val_total_loss'
        }]

    def get_scheduler(self, optimiser: Optional[torch.optim.Adam]):
        num_train_steps = int(self.params['train_set_len'] / self.params['batch_size']) * self.params['max_epochs']
        num_warmup_steps = int(num_train_steps * self.params['warmup_proportion'])
        return get_linear_schedule_with_warmup(optimizer=optimiser, num_training_steps=num_train_steps,
                                               num_warmup_steps=num_warmup_steps)
