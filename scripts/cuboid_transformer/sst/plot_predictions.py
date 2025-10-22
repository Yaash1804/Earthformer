# ==============================================================================
# SCRIPT FOR PLOTTING MODEL PREDICTIONS vs. ACTUAL DATA (FINAL CORRECTED VERSION)
#
# This version now calculates and displays the MSE for each data split on the plot.
#
# INSTRUCTIONS:
# 1. Ensure `sst_datamodule.py` has the small addition for the time index.
# 2. Save this file as `plot_predictions.py`.
# 3. Update the `CHECKPOINT_PATH` and `save_directory_name` variables below.
# 4. Run from a Colab cell using the command:
#    !pip install scikit-learn
#    !MPLBACKEND=Agg python /path/to/your/plot_predictions.py
# ==============================================================================

# --- ROBUST FIX for Matplotlib Backend Error ---
import matplotlib
matplotlib.use('Agg')
# --- END OF FIX ---

import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
from omegaconf import OmegaConf
import warnings
# --- NEW IMPORT FOR MSE CALCULATION ---
from sklearn.metrics import mean_squared_error

warnings.filterwarnings("ignore", "This DataLoader will create .* worker processes .*")
warnings.filterwarnings("ignore", "torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.")

from src.earthformer.datasets.sst.sst_datamodule import SSTDataModule
from train_cuboid_sst import CuboidSSTPLModule


def plot_model_predictions():
    """
    Main function to load the model, generate predictions, and create a comparison plot.
    """
    # --- 1. CONFIGURATION ---
    print("Step 1: Configuring paths...")
    
    CFG_PATH = "/content/Earthformer/scripts/cuboid_transformer/sst/sst.yaml"
    
    # !!! UPDATE THIS PATH !!!
    CHECKPOINT_PATH = "/content/Earthformer/scripts/cuboid_transformer/sst/experiments/sst_colab_run_1/checkpoints/model-epoch=019.ckpt"

    # !!! UPDATE THIS NAME !!!
    save_directory_name = "sst_colab_run_1"
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint file not found at {CHECKPOINT_PATH}. Please update the path.")

    # --- 2. LOAD TRAINED MODEL ---
    print("\nStep 2: Loading trained model from checkpoint...")
    
    model = CuboidSSTPLModule.load_from_checkpoint(
        CHECKPOINT_PATH, 
        oc_file=CFG_PATH, 
        save_dir=save_directory_name
    )
    
    model.to(DEVICE)
    model.eval()
    print("Model loaded successfully.")

    # --- 3. PREPARE THE DATA ---
    print("\nStep 3: Setting up the datamodule using the model's saved configuration...")

    dataset_cfg = OmegaConf.to_object(model.hparams.dataset)
    dataset_cfg.pop("_target_", None)
    for key in ["train_start_year", "val_start_year", "test_start_year", "end_year"]:
        dataset_cfg.pop(key, None)

    datamodule = SSTDataModule(**dataset_cfg)
    datamodule.setup()
    
    # Create a non-shuffled train_loader for chronological plotting
    train_loader = torch.utils.data.DataLoader(
        datamodule.sst_train,
        batch_size=datamodule.hparams.batch_size,
        shuffle=False, # Use False for chronological plotting
        num_workers=datamodule.hparams.num_workers
    )
    val_loader = datamodule.val_dataloader()
    test_loader = datamodule.test_dataloader()
    print("Data is ready.")
    
    # --- 4. INFERENCE FUNCTION ---
    def get_onestep_forecasts(loader, model):
        all_preds_onestep = []
        all_actuals_onestep = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                print(f"  Processing batch {i+1}/{len(loader)}...", end='\r')
                x, y = batch
                x = x.to(DEVICE)
                pred_y = model.torch_nn_module(x.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
                avg_preds = pred_y.mean(dim=[3, 4])
                avg_actuals = y.mean(dim=[3, 4])
                all_preds_onestep.append(avg_preds[:, 0].cpu().numpy())
                all_actuals_onestep.append(avg_actuals[:, 0].cpu().numpy())
        print("\n")
        return np.concatenate(all_preds_onestep), np.concatenate(all_actuals_onestep)

    # --- 5. GENERATE PREDICTIONS ---
    print("\nStep 5: Generating 1-step-ahead forecasts...")
    print("Processing training data...")
    train_preds_norm, train_actuals_norm = get_onestep_forecasts(train_loader, model)
    print("Processing validation data...")
    val_preds_norm, val_actuals_norm = get_onestep_forecasts(val_loader, model)
    print("Processing test data...")
    test_preds_norm, test_actuals_norm = get_onestep_forecasts(test_loader, model)
    print("Forecast generation complete.")

    # --- 6. DE-NORMALIZE DATA ---
    print("\nStep 6: De-normalizing data to actual SST values (°C)...")
    mean = datamodule.mean
    std = datamodule.std
    
    train_preds = (train_preds_norm * std) + mean
    train_actuals = (train_actuals_norm * std) + mean
    val_preds = (val_preds_norm * std) + mean
    val_actuals = (val_actuals_norm * std) + mean
    test_preds = (test_preds_norm * std) + mean
    test_actuals = (test_actuals_norm * std) + mean
    print("De-normalization complete.")

    # --- NEW: CALCULATE MSE FOR EACH SPLIT ---
    mse_train = mean_squared_error(train_actuals, train_preds)
    mse_val = mean_squared_error(val_actuals, val_preds)
    mse_test = mean_squared_error(test_actuals, test_preds)
    print(f"\nCalculated MSEs - Train: {mse_train:.4f}, Val: {mse_val:.4f}, Test: {mse_test:.4f}")
    # --- END OF NEW SECTION ---

    # --- 7. GET TIME AXIS ---
    print("\nStep 7: Preparing time coordinates for plotting...")
    in_len = datamodule.hparams.in_len
    train_times = datamodule.train_time_index[in_len : len(datamodule.sst_train) + in_len]
    val_times = datamodule.val_time_index[in_len : len(datamodule.sst_val) + in_len]
    test_times = datamodule.test_time_index[in_len : len(datamodule.sst_test) + in_len]
    
    # --- 8. CREATE THE PLOT ---
    print("\nStep 8: Creating the plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(20, 8))

    # Plot Actual Data
    ax.plot(train_times, train_actuals, label='Actual Train', color='blue', linewidth=2)
    ax.plot(val_times, val_actuals, label='Actual Val', color='green', linewidth=2)
    ax.plot(test_times, test_actuals, label='Actual Test', color='red', linewidth=2)

    # Plot Predicted Data
    ax.plot(train_times, train_preds, label='Predicted Train (1-Month Ahead)', linestyle='--', color='cyan', alpha=0.9)
    ax.plot(val_times, val_preds, label='Predicted Val (1-Month Ahead)', linestyle='--', color='lime', alpha=0.9)
    ax.plot(test_times, test_preds, label='Predicted Test (1-Month Ahead)', linestyle='--', color='magenta', alpha=0.9)

    # --- MODIFICATION: UPDATED TITLE WITH MSE VALUES ---
    ax.set_title(f'SST Forecasting: Actual vs. Predicted\n'
                 f'MSE - Train: {mse_train:.4f} | Val: {mse_val:.4f} | Test: {mse_test:.4f}',
                 fontsize=16)
    # --- END OF MODIFICATION ---

    # Formatting
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Spatially-Averaged SST (°C)', fontsize=14)
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True)
    fig.tight_layout()

    save_path = 'sst_onestep_predictions_best_model.png'
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot saved successfully as {save_path}")

if __name__ == '__main__':
    plot_model_predictions()
