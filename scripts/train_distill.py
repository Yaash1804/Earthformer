import os
import argparse
import warnings
import torch
import lightning as L
from omegaconf import OmegaConf

# --- Assumed/Required Imports from Earthformer Repo ---
# This assumes the SEVIR DataModule is defined in this path.
# Adjust if your patch data uses a different DataModule.
try:
    from earthformer.datamodules.sevir import SevirDataModule
except ImportError:
    print("WARNING: Could not import 'SevirDataModule' from 'earthformer.datamodules.sevir'.")
    print("Please ensure the DataModule is in the correct path.")
    # Define a placeholder
    class SevirDataModule(L.LightningDataModule):
        def __init__(self, *args, **kwargs):
            super().__init__()
            print("ERROR: Using placeholder SevirDataModule!")
            raise NotImplementedError("Original SevirDataModule not found!")
# ----------------------------------------------------

# Import the new LightningModule
from earthformer.distill_lightning import DistillLitModule

def main(args):
    # Load configuration file
    print(f"Loading experiment configuration from: {args.config_path}")
    cfg = OmegaConf.load(args.config_path)

    # Set up data module
    # The config file (sevir_distill.yaml) must have a 'datamodule' section
    print("Setting up DataModule...")
    datamodule_cfg = cfg.get('datamodule', {})
    if not datamodule_cfg:
        raise ValueError("Config file must contain a 'datamodule' section.")
        
    # This assumes you are using a DataModule compatible with the SEVIR one,
    # which takes 'seq_len', 'pred_len', 'batch_size', etc.
    # Ensure your datamodule_cfg provides the correct 12-in, 12-out sequence
    datamodule = SevirDataModule(**datamodule_cfg)

    # Set up the Teacher-Student Lightning Module
    print("Setting up DistillLitModule...")
    student_cfg = cfg.get('student', None)
    optimizer_cfg = cfg.get('optimizer', None)
    if not student_cfg or not optimizer_cfg:
        raise ValueError("Config file must contain 'student' and 'optimizer' sections.")

    model = DistillLitModule(
        teacher_ckpt_path=args.teacher_ckpt_path,
        teacher_config_path=args.teacher_config_path,
        student_cfg=student_cfg,
        optimizer_cfg=optimizer_cfg
    )

    # Set up the Trainer
    # The config file must have a 'trainer' section
    print("Setting up Trainer...")
    trainer_cfg = cfg.get('trainer', {})
    if not trainer_cfg:
        raise ValueError("Config file must contain a 'trainer' section.")

    # Setup callbacks (e.g., ModelCheckpoint)
    callbacks =
    if 'callbacks' in trainer_cfg:
        for cb_name, cb_cfg in trainer_cfg.callbacks.items():
            if cb_name == 'model_checkpoint':
                print("Setting up ModelCheckpoint callback.")
                cb_cfg_dict = OmegaConf.to_container(cb_cfg, resolve=True)
                callbacks.append(L.pytorch.callbacks.ModelCheckpoint(**cb_cfg_dict))
            # Add other callbacks like EarlyStopping if needed
            elif cb_name == 'early_stopping':
                print("Setting up EarlyStopping callback.")
                cb_cfg_dict = OmegaConf.to_container(cb_cfg, resolve=True)
                callbacks.append(L.pytorch.callbacks.EarlyStopping(**cb_cfg_dict))

    # Clean trainer_cfg from callback definitions if they exist
    trainer_cfg_dict = OmegaConf.to_container(trainer_cfg, resolve=True)
    trainer_cfg_dict.pop('callbacks', None)

    trainer = L.Trainer(
        callbacks=callbacks,
        **trainer_cfg_dict
    )

    # Start training
    print("Starting training...")
    trainer.fit(model, datamodule=datamodule)

    print("Training finished.")
    
    # Optionally, run testing
    if args.test_after_train:
        print("Starting testing...")
        trainer.test(model, datamodule=datamodule, ckpt_path='best')
        print("Testing finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Earthformer-ConvLSTM Refinement Model")
    
    parser.add_argument('--config_path', type=str, required=True,
                        help="Path to the.yaml config file for the distillation experiment.")
                        
    parser.add_qrguemnt('--teacher_ckpt_path', type=str, required=True,
                        help="Path to the pre-trained Earthformer teacher model checkpoint (.ckpt).")
                        
    parser.add_argument('--teacher_config_path', type=str, required=True,
                        help="Path to the original.yaml config file used to train the teacher model.")
                        
    parser.add_argument('--test_after_train', action='store_true',
                        help="Run the test set after training is complete.")

    args = parser.parse_args()
    main(args)
