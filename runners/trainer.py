import os
from typing import Dict

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, LightningLoggerBase


def get_tensorboard_logger(log_dir: str) -> LightningLoggerBase:
    return TensorBoardLogger(save_dir=log_dir, name="tensorboard_logs")


def get_trainer(params: Dict) -> Trainer:
    mode = 'min' if 'loss' in params['monitor_quantity'] else 'max'
    model_checkpoint = ModelCheckpoint(
        monitor=params['monitor_quantity'],
        mode=mode,
        filename="{epoch:02d}-{" + str(params['monitor_quantity']) + ":.3f}",
        save_last=True,
        save_top_k=1,
        dirpath=os.path.join(params['output_dir'], 'checkpoints'),
    )

    early_stop_callback = EarlyStopping(monitor=params['monitor_quantity'],
                                        patience=params['patience'],
                                        mode=mode)
    trainer = Trainer(gpus=-1 if torch.cuda.device_count() >= 1 else None,
                      max_epochs=params['max_epochs'],
                      logger=get_tensorboard_logger(params['output_dir']),
                      callbacks=[model_checkpoint, early_stop_callback],
                      gradient_clip_val=params['grad_clip_val'],
                      checkpoint_callback=True,
                      progress_bar_refresh_rate=10,
                      weights_summary=None,
                      accelerator='ddp' if torch.cuda.device_count() > 1 else None)
    return trainer
