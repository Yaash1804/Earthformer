# In file: scripts/cuboid_transformer/sst/train_cuboid_sst.py

import warnings
from shutil import copyfile
import inspect
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from omegaconf import OmegaConf
import os
import argparse

# --- MODIFICATION START ---
# Import the custom SSTDataModule instead of the ENSO-specific one
from src.earthformer.datasets.sst.sst_datamodule import SSTDataModule
# --- MODIFICATION END ---

from earthformer.config import cfg
from earthformer.utils.optim import SequentialLR, warmup_lambda
from earthformer.utils.utils import get_parameter_names
from earthformer.cuboid_transformer.cuboid_transformer import CuboidTransformerModel
from earthformer.utils.apex_ddp import ApexDDPStrategy


_curr_dir = os.path.realpath(os.path.dirname(os.path.realpath(__file__)))
exps_dir = os.path.join(_curr_dir, "experiments")

# --- MODIFICATION START ---
# Renamed the main PyTorch Lightning class to be specific to our SST task
class CuboidSSTPLModule(pl.LightningModule):
# --- MODIFICATION END ---

    def __init__(self,
                 total_num_steps: int,
                 oc_file: str = None,
                 save_dir: str = None):
        super().__init__()
        if oc_file is not None:
            oc_from_file = OmegaConf.load(open(oc_file, "r"))
        else:
            oc_from_file = None
        oc = self.get_base_config(oc_from_file=oc_from_file)
        model_cfg = OmegaConf.to_object(oc.model)
        
        # This part for setting up the model architecture remains largely the same
        num_blocks = len(model_cfg["enc_depth"])
        if isinstance(model_cfg["self_pattern"], str):
            enc_attn_patterns = [model_cfg["self_pattern"]] * num_blocks
        else:
            enc_attn_patterns = OmegaConf.to_container(model_cfg["self_pattern"])
        if isinstance(model_cfg["cross_self_pattern"], str):
            dec_self_attn_patterns = [model_cfg["cross_self_pattern"]] * num_blocks
        else:
            dec_self_attn_patterns = OmegaConf.to_container(model_cfg["cross_self_pattern"])
        if isinstance(model_cfg["cross_pattern"], str):
            dec_cross_attn_patterns = [model_cfg["cross_pattern"]] * num_blocks
        else:
            dec_cross_attn_patterns = OmegaConf.to_container(model_cfg["cross_pattern"])

        self.torch_nn_module = CuboidTransformerModel(
            input_shape=model_cfg["input_shape"],
            target_shape=model_cfg["target_shape"],
            base_units=model_cfg["base_units"],
            block_units=model_cfg["block_units"],
            scale_alpha=model_cfg["scale_alpha"],
            enc_depth=model_cfg["enc_depth"],
            dec_depth=model_cfg["dec_depth"],
            enc_use_inter_ffn=model_cfg["enc_use_inter_ffn"],
            dec_use_inter_ffn=model_cfg["dec_use_inter_ffn"],
            dec_hierarchical_pos_embed=model_cfg["dec_hierarchical_pos_embed"],
            downsample=model_cfg["downsample"],
            downsample_type=model_cfg["downsample_type"],
            enc_attn_patterns=enc_attn_patterns,
            dec_self_attn_patterns=dec_self_attn_patterns,
            dec_cross_attn_patterns=dec_cross_attn_patterns,
            num_heads=model_cfg["num_heads"],
            attn_drop=model_cfg["attn_drop"],
            proj_drop=model_cfg["proj_drop"],
            ffn_drop=model_cfg["ffn_drop"],
            upsample_type=model_cfg["upsample_type"],
            ffn_activation=model_cfg["ffn_activation"],
            gated_ffn=model_cfg["gated_ffn"],
            norm_layer=model_cfg["norm_layer"],
            padding_type=model_cfg["padding_type"],
            checkpoint_level=model_cfg["checkpoint_level"],
        )

        self.total_num_steps = total_num_steps
        self.save_hyperparameters(oc)
        self.oc = oc
        
        self.in_len = oc.layout.in_len
        self.out_len = oc.layout.out_len
        self.layout = oc.layout.layout
        self.channel_axis = self.layout.find("C")
        self.batch_axis = self.layout.find("N")
        
        self.save_dir = save_dir

        self.valid_mse = torchmetrics.MeanSquaredError()
        self.valid_mae = torchmetrics.MeanAbsoluteError()
        self.test_mse = torchmetrics.MeanSquaredError()
        self.test_mae = torchmetrics.MeanAbsoluteError()

        self.configure_save(cfg_file_path=oc_file)

    def configure_save(self, cfg_file_path=None):
        self.save_dir = os.path.join(exps_dir, self.save_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        if cfg_file_path is not None:
            cfg_file_target_path = os.path.join(self.save_dir, "cfg.yaml")
            if not os.path.exists(cfg_file_target_path) or \
               not os.path.samefile(cfg_file_path, cfg_file_target_path):
                copyfile(cfg_file_path, cfg_file_target_path)

    def get_base_config(self, oc_from_file=None):
        oc = OmegaConf.create()
        oc.layout = self.get_layout_config()
        oc.optim = self.get_optim_config()
        oc.logging = self.get_logging_config()
        oc.trainer = self.get_trainer_config()
        oc.model = self.get_model_config()
        oc.dataset = self.get_dataset_config()
        if oc_from_file is not None:
            oc = OmegaConf.merge(oc, oc_from_file)
        return oc

    @staticmethod
    def get_layout_config():
        cfg = OmegaConf.create()
        cfg.in_len = 12
        cfg.out_len = 12
        # --- MODIFICATION START ---
        # Updated to match the SST dataset dimensions
        cfg.img_height = 720
        cfg.img_width = 1440
        # The layout is now NTHWC (Batch, Time, Height, Width, Channel)
        cfg.layout = "NTHWC"
        # --- MODIFICATION END ---
        return cfg

    @classmethod
    def get_model_config(cls):
        cfg = OmegaConf.create()
        layout_cfg = cls.get_layout_config()
        # --- MODIFICATION START ---
        # Updated to 1 channel for SST data
        cfg.data_channels = 1
        cfg.input_shape = (layout_cfg.in_len, layout_cfg.img_height, layout_cfg.img_width, cfg.data_channels)
        cfg.target_shape = (layout_cfg.out_len, layout_cfg.img_height, layout_cfg.img_width, cfg.data_channels)
        # --- MODIFICATION END ---

        cfg.base_units = 64
        cfg.block_units = None
        cfg.scale_alpha = 1.0
        cfg.enc_depth = [1, 1]
        cfg.dec_depth = [1, 1]
        cfg.enc_use_inter_ffn = True
        cfg.dec_use_inter_ffn = True
        cfg.dec_hierarchical_pos_embed = True
        cfg.downsample = 2
        cfg.downsample_type = "patch_merge"
        cfg.upsample_type = "upsample"
        cfg.self_pattern = 'axial'
        cfg.cross_self_pattern = 'axial'
        cfg.cross_pattern = 'cross_1x1'
        cfg.attn_drop = 0.1
        cfg.proj_drop = 0.1
        cfg.ffn_drop = 0.1
        cfg.num_heads = 4
        cfg.ffn_activation = 'gelu'
        cfg.gated_ffn = False
        cfg.norm_layer = 'layer_norm'
        cfg.padding_type = 'zeros'
        cfg.checkpoint_level = 2
        return cfg

    @classmethod
    def get_dataset_config(cls):
        cfg = OmegaConf.create()
        # --- MODIFICATION START ---
        # These parameters correspond to the arguments in our SSTDataModule
        cfg.in_len = 12
        cfg.out_len = 12
        cfg.batch_size = 4
        cfg.num_workers = 4
        cfg.train_start_year = 1981
        cfg.val_start_year = 2016
        cfg.test_start_year = 2021
        cfg.end_year = 2025
        # --- MODIFICATION END ---
        return cfg

    @staticmethod
    def get_optim_config():
        cfg = OmegaConf.create()
        cfg.seed = 0
        cfg.total_batch_size = 32
        cfg.micro_batch_size = 8
        cfg.method = "adamw"
        cfg.lr = 1E-4
        cfg.wd = 1E-5
        cfg.gradient_clip_val = 1.0
        cfg.max_epochs = 200
        cfg.warmup_percentage = 0.2
        cfg.lr_scheduler_mode = "cosine"
        cfg.min_lr_ratio = 0.1
        cfg.warmup_min_lr_ratio = 0.1
        cfg.save_top_k = 1
        return cfg

    @staticmethod
    def get_logging_config():
        cfg = OmegaConf.create()
        # --- MODIFICATION START ---
        # Changed logging prefix to be specific to SST
        cfg.logging_prefix = "SST_Forecasting"
        # --- MODIFICATION END ---
        cfg.monitor_lr = True
        cfg.monitor_device = False
        return cfg

    @staticmethod
    def get_trainer_config():
        cfg = OmegaConf.create()
        cfg.check_val_every_n_epoch = 1
        cfg.log_step_ratio = 0.01
        cfg.precision = 16
        return cfg

    def configure_optimizers(self):
        decay_parameters = get_parameter_names(self.torch_nn_module, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.torch_nn_module.named_parameters() if n in decay_parameters],
             'weight_decay': self.oc.optim.wd},
            {'params': [p for n, p in self.torch_nn_module.named_parameters() if n not in decay_parameters],
             'weight_decay': 0.0}
        ]
        optimizer = torch.optim.AdamW(params=optimizer_grouped_parameters, lr=self.oc.optim.lr, weight_decay=self.oc.optim.wd)
        warmup_iter = int(np.round(self.oc.optim.warmup_percentage * self.total_num_steps))
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup_lambda(warmup_steps=warmup_iter, min_lr_ratio=self.oc.optim.warmup_min_lr_ratio))
        cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(self.total_num_steps - warmup_iter), eta_min=self.oc.optim.min_lr_ratio * self.oc.optim.lr)
        lr_scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_iter])
        lr_scheduler_config = {'scheduler': lr_scheduler, 'interval': 'step', 'frequency': 1}
        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler_config}

    def set_trainer_kwargs(self, **kwargs):
        checkpoint_callback = ModelCheckpoint(
            monitor="valid_mse_epoch",
            dirpath=os.path.join(self.save_dir, "checkpoints"),
            filename="model-{epoch:03d}",
            save_top_k=self.oc.optim.save_top_k,
            save_last=True,
            mode="min",
        )
        callbacks = kwargs.pop("callbacks",)
        callbacks += [checkpoint_callback]
        if self.oc.logging.monitor_lr:
            callbacks +=
        
        log_every_n_steps = max(1, int(self.oc.trainer.log_step_ratio * self.total_num_steps))
        
        ret = dict(
            callbacks=callbacks,
            log_every_n_steps=log_every_n_steps,
            default_root_dir=self.save_dir,
            accelerator="gpu",
            strategy=ApexDDPStrategy(find_unused_parameters=False, delay_allreduce=True),
            max_epochs=self.oc.optim.max_epochs,
            check_val_every_n_epoch=self.oc.trainer.check_val_every_n_epoch,
            gradient_clip_val=self.oc.optim.gradient_clip_val,
            precision=self.oc.trainer.precision,
        )
        ret.update(kwargs)
        return ret

    # --- MODIFICATION START ---
    # Simplified forward pass for a standard forecasting task
    def forward(self, batch):
        x, y = batch
        # The model expects a 5D tensor: (N, T, H, W, C)
        # Our dataloader provides (N, T, C, H, W), so we permute the dimensions
        x = x.permute(0, 1, 3, 4, 2)
        y = y.permute(0, 1, 3, 4, 2)
        
        pred_y = self.torch_nn_module(x)
        loss = F.mse_loss(pred_y, y)
        return pred_y, loss, x, y
    # --- MODIFICATION END ---

    def training_step(self, batch, batch_idx):
        _, loss, _, _ = self(batch)
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    # --- MODIFICATION START ---
    # Simplified validation step with standard metrics
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        pred_seq, _, _, target_seq = self(batch)
        if self.oc.trainer.precision == 16:
            pred_seq = pred_seq.float()
        self.valid_mse(pred_seq, target_seq)
        self.valid_mae(pred_seq, target_seq)
    # --- MODIFICATION END ---

    # --- MODIFICATION START ---
    # Simplified validation end step, logging only MSE and MAE
    def validation_epoch_end(self, outputs):
        valid_mse = self.valid_mse.compute()
        valid_mae = self.valid_mae.compute()
        self.log('valid_mse_epoch', valid_mse, prog_bar=True, on_step=False, on_epoch=True)
        self.log('valid_mae_epoch', valid_mae, prog_bar=True, on_step=False, on_epoch=True)
        self.valid_mse.reset()
        self.valid_mae.reset()
    # --- MODIFICATION END ---

    # --- MODIFICATION START ---
    # Simplified test step with standard metrics
    def test_step(self, batch, batch_idx, dataloader_idx=0):
        pred_seq, _, _, target_seq = self(batch)
        if self.oc.trainer.precision == 16:
            pred_seq = pred_seq.float()
        self.test_mse(pred_seq, target_seq)
        self.test_mae(pred_seq, target_seq)
    # --- MODIFICATION END ---

    # --- MODIFICATION START ---
    # Simplified test end step, logging only MSE and MAE
    def test_epoch_end(self, outputs):
        test_mse = self.test_mse.compute()
        test_mae = self.test_mae.compute()
        self.log('test_mse_epoch', test_mse, prog_bar=True, on_step=False, on_epoch=True)
        self.log('test_mae_epoch', test_mae, prog_bar=True, on_step=False, on_epoch=True)
        self.test_mse.reset()
        self.test_mae.reset()
    # --- MODIFICATION END ---

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save', default='tmp_sst', type=str)
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--cfg', default=None, type=str, help="Path to YAML config file.")
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--ckpt_name', default=None, type=str, help='The model checkpoint name.')
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.cfg is not None:
        oc_from_file = OmegaConf.load(open(args.cfg, "r"))
        dataset_cfg = OmegaConf.to_object(oc_from_file.dataset)
        total_batch_size = oc_from_file.optim.total_batch_size
        micro_batch_size = oc_from_file.optim.micro_batch_size
        max_epochs = oc_from_file.optim.max_epochs
        seed = oc_from_file.optim.seed
    else:
        # Fallback to default configs if no YAML is provided
        dataset_cfg = OmegaConf.to_object(CuboidSSTPLModule.get_dataset_config())
        optim_cfg = OmegaConf.to_object(CuboidSSTPLModule.get_optim_config())
        micro_batch_size = optim_cfg['micro_batch_size']
        total_batch_size = int(micro_batch_size * args.gpus)
        max_epochs = optim_cfg['max_epochs']
        seed = optim_cfg['seed']
    
    seed_everything(seed, workers=True)
    
    # --- MODIFICATION START ---
    # Instantiate our custom SSTDataModule
    dm = SSTDataModule(
        data_root=cfg.datasets_dir, # This can be overridden by the YAML file
        **dataset_cfg
    )
    # --- MODIFICATION END ---
    
    dm.prepare_data()
    dm.setup()
    
    accumulate_grad_batches = total_batch_size // (micro_batch_size * args.gpus)
    
    # Calculate total steps
    num_train_samples = len(dm.sst_train)
    total_num_steps = int(max_epochs * num_train_samples / total_batch_size)
    
    pl_module = CuboidSSTPLModule(
        total_num_steps=total_num_steps,
        save_dir=args.save,
        oc_file=args.cfg
    )
    
    trainer_kwargs = pl_module.set_trainer_kwargs(
        devices=args.gpus,
        accumulate_grad_batches=accumulate_grad_batches,
    )
    trainer = Trainer(**trainer_kwargs)

    if args.test:
        assert args.ckpt_name is not None, "ckpt_name must be provided for testing."
        ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", args.ckpt_name)
        trainer.test(model=pl_module, datamodule=dm, ckpt_path=ckpt_path)
    else:
        ckpt_path = None
        if args.ckpt_name is not None:
            ckpt_path = os.path.join(pl_module.save_dir, "checkpoints", args.ckpt_name)
            if not os.path.exists(ckpt_path):
                warnings.warn(f"Checkpoint {ckpt_path} not found! Starting training from scratch.")
                ckpt_path = None
        
        trainer.fit(model=pl_module, datamodule=dm, ckpt_path=ckpt_path)
        trainer.test(model=pl_module, datamodule=dm)

if __name__ == "__main__":
    main()