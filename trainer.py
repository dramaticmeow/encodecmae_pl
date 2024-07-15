import pytorch_lightning as pl
from omegaconf import OmegaConf
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import TQDMProgressBar, LearningRateMonitor, ModelCheckpoint

from models.builder import build_EncodecMAE
from datasets import EncodecMAEDataModule
import os, logging
RUN = 'encodecmae_base'
print(f'Running: {RUN}')

PROJECT_NAME = 'encodecMAE'
MODEL_CKPT = None #'nowcqbo0/checkpoints/epoch=0-step=112000.ckpt'
WANDB_RESUME =  None # 'nowcqbo0' #'uuu83nkt' #'r4x28884'
TRAIN_CKPT =  None #f'{WANDB_RESUME}/checkpoints/epoch=0-step=112000.ckpt'

import os

class GradNormCallback(Callback):
    """
    Logs the gradient norm.
    """

    def on_before_optimizer_step(self, trainer, model, optimizer):
        model.log("train/grad_norm", gradient_norm(model), prog_bar=True, on_step=True, on_epoch=False, logger=True)

def gradient_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm


def main():
    
    train_args = OmegaConf.load(f'./config/{RUN}.yaml')
    seed = train_args.get('seed', 1465)
    pl.seed_everything(seed)
    print(f'Seed: {seed}')

    model = build_EncodecMAE(train_args)

    # # Train the model
    # if MODEL_CKPT is not None:
    #     model = model.load_from_checkpoint(f'./encodecMAE/{MODEL_CKPT}', map_location='cpu')
    
    wandb_logger = WandbLogger(project=PROJECT_NAME, name=RUN, id=WANDB_RESUME)


    trainer = pl.Trainer(
        accelerator="auto", strategy="ddp_find_unused_parameters_true", num_nodes=1,
        max_steps=train_args.total_steps,
        accumulate_grad_batches=train_args.dataset.grad_acc,
        num_sanity_val_steps=1,
        precision='bf16-mixed',
        logger=wandb_logger,
        val_check_interval=train_args.ckpt_interval,
        check_val_every_n_epoch=None,
        # gradient_clip_val=1.0,
        # plugins=[MyClusterEnvironment()],
        callbacks=[GradNormCallback(), LearningRateMonitor(), ModelCheckpoint(dirpath=f'./encodecMAE/{RUN}', every_n_train_steps=train_args.ckpt_interval, save_top_k=-1)],
        # barebones=True,
        # profiler='simple',
        # enable_progress_bar=False
    )

    dm = EncodecMAEDataModule(train_args)
    dm.setup()
    if TRAIN_CKPT is not None:
        trainer.fit(model, datamodule=dm, ckpt_path=f'./encodecMAE/{TRAIN_CKPT}')
    else:
        trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()