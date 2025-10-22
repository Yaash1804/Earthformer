# ==============================================================================
# SCRIPT FOR COMPARING ONE-STEP-AHEAD vs. AUTOREGRESSIVE FORECASTING (VERIFIED)
#
# This version fixes the AxisError by correcting the dimensions used for
# averaging the actual data.
#
# INSTRUCTIONS:
# 1. Save this file as `compare_forecast_methods.py`.
# 2. Update the CHECKPOINT_PATH and DATA_ROOT variables below.
# 3. Run from a Colab cell using the command:
#    !pip install scikit-learn
#    !MPLBACKEND=Agg python /path/to/your/compare_forecast_methods.py
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
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# Suppress common warnings for cleaner output
warnings.filterwarnings("ignore", "This DataLoader will create .* worker processes .*")
warnings.filterwarnings("ignore", "torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.")

from train_cuboid_sst import CuboidSSTPLModule

def generate_onestep_forecasts(model, full_data_normalized, hparams):
    """
    METHOD 1: Generates one-step-ahead forecasts using a sliding window of actual data.
    """
    print("\n===== Generating One-Step-Ahead Forecasts =====")
    all_preds_norm = []
    in_len = hparams.dataset.in_len
    
    input_seqs = []
    # The number of sequences is total length minus the length of one full sequence (in+out) + 1
    # But since we only predict one step, we just need to ensure we have `in_len` history.
    for i in range(len(full_data_normalized) - in_len):
        input_seqs.append(full_data_normalized[i:i + in_len])
    input_seqs = torch.from_numpy(np.array(input_seqs)).float()

    with torch.no_grad():
        batch_size = hparams.dataset.batch_size
        num_batches = -(-len(input_seqs) // batch_size)
        for i in tqdm(range(0, len(input_seqs), batch_size), desc="  Processing batches"):
            x_batch = input_seqs[i:i + batch_size].to(model.device)
            
            pred_y_batch = model.torch_nn_module(x_batch.permute(0, 1, 3, 4, 2))
            pred_y_batch = pred_y_batch.permute(0, 1, 4, 2, 3)
            avg_preds = pred_y_batch.mean(dim=[3, 4])
            all_preds_norm.append(avg_preds[:, 0].cpu().numpy())
            
    return np.concatenate(all_preds_norm)

def generate_autoregressive_forecasts(model, full_data_normalized, hparams):
    """
    METHOD 2: Generates forecasts autoregressively, feeding predictions back as input.
    """
    print("\n===== Generating Autoregressive Forecasts =====")
    autoregressive_preds_norm = []
    in_len = hparams.dataset.in_len
    
    current_window = torch.from_numpy(full_data_normalized[:in_len]).float()
    
    num_predictions = len(full_data_normalized) - in_len
    with torch.no_grad():
        for i in tqdm(range(num_predictions), desc="  Forecasting step-by-step"):
            input_tensor = current_window.unsqueeze(0).to(model.device)
            
            pred_y = model.torch_nn_module(input_tensor.permute(0, 1, 3, 4, 2))
            pred_y = pred_y.permute(0, 1, 4, 2, 3) # Shape: (1, 12, 1, 21, 28)

            next_step_map = pred_y[:, 0, :, :, :] 
            avg_for_storage = next_step_map.mean().item()
            autoregressive_preds_norm.append(avg_for_storage)
            
            current_window = torch.cat([current_window[1:], next_step_map.cpu()], dim=0)

    return np.array(autoregressive_preds_norm)

def main():
    """
    Main function to load model once, run both forecasting methods, and plot results.
    """
    # --- 1. CONFIGURATION ---
    CFG_PATH = "/content/Earthformer/scripts/cuboid_transformer/sst/sst.yaml"
    DATA_ROOT = "/content/drive/MyDrive/sst"
    CHECKPOINT_PATH = "/content/Earthformer/scripts/cuboid_transformer/sst/experiments/sst_colab_run_1/checkpoints/model-epoch=019.ckpt"
    save_directory_name = "sst_colab_run_1"

    # --- 2. LOAD MODEL AND DATA (Done once) ---
    print("Loading model and data...")
    model = CuboidSSTPLModule.load_from_checkpoint(CHECKPOINT_PATH, oc_file=CFG_PATH, save_dir=save_directory_name)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()

    hparams = model.hparams
    file_path = os.path.join(DATA_ROOT, "sst.mon.mean.nc")
    ds_full = xr.open_dataset(file_path)

    # --- 3. PREPARE FULL DATASET & NORMALIZATION STATS ---
    print("Preparing full dataset and calculating original normalization stats...")
    original_train_end_year = hparams.dataset.get('train_end_year', 2015)
    original_lat_slice = slice(15.625, 20.625)
    original_lon_slice = slice(65.625, 72.375)
    original_train_slice = slice(None, str(original_train_end_year))
    
    train_data_for_stats = ds_full['sst'].sel(time=original_train_slice, lat=original_lat_slice, lon=original_lon_slice).values.astype(np.float32)
    mean_original = np.nanmean(train_data_for_stats)
    std_original = np.nanstd(train_data_for_stats)
    print(f"  Original Training Mean: {mean_original:.4f}, Original Training Std: {std_original:.4f}")

    # Create one continuous array for the entire dataset
    full_data_raw = ds_full.sel(lat=original_lat_slice, lon=original_lon_slice)['sst'].values.astype(np.float32)
    full_data_filled = np.nan_to_num(full_data_raw, nan=mean_original)
    full_data_normalized = (full_data_filled - mean_original) / std_original
    full_data_normalized = full_data_normalized[:, np.newaxis, :, :]
    
    full_time_index = ds_full.sel(lat=original_lat_slice, lon=original_lon_slice).get_index("time")
    in_len = hparams.dataset.in_len
    
    # --- 4. RUN BOTH FORECASTING METHODS ---
    onestep_preds_norm = generate_onestep_forecasts(model, full_data_normalized, hparams)
    autoregressive_preds_norm = generate_autoregressive_forecasts(model, full_data_normalized, hparams)
    
    # --- 5. PREPARE DATA FOR PLOTTING ---
    print("\nDe-normalizing data and calculating MSEs...")
    
    onestep_preds_denorm = (onestep_preds_norm * std_original) + mean_original
    autoregressive_preds_denorm = (autoregressive_preds_norm * std_original) + mean_original
    
    # Get the slice of actual data that corresponds to the predictions
    num_preds = len(onestep_preds_denorm)
    actual_data_for_comparison = full_data_filled[in_len : in_len + num_preds]
    
    # --- THIS IS THE CORRECTED LINE ---
    # The array has 3 dimensions (Time, Height, Width), so we average over axes 1 and 2.
    actual_data_avg = actual_data_for_comparison.mean(axis=(1, 2))
    # --- END OF CORRECTION ---

    time_axis = full_time_index[in_len : in_len + num_preds]
    
    # Calculate MSEs
    mse_onestep = mean_squared_error(actual_data_avg, onestep_preds_denorm)
    mse_autoregressive = mean_squared_error(actual_data_avg, autoregressive_preds_denorm)
    print(f"  One-Step-Ahead MSE: {mse_onestep:.4f}")
    print(f"  Autoregressive MSE: {mse_autoregressive:.4f}")

    # --- 6. CREATE PLOTS ---
    # Plot 1: One-Step-Ahead Forecast
    print("\nCreating Plot 1: One-Step-Ahead Forecast...")
    fig1, ax1 = plt.subplots(figsize=(20, 8))
    ax1.plot(time_axis, actual_data_avg, label='Actual SST', color='blue', linewidth=2)
    ax1.plot(time_axis, onestep_preds_denorm, label='Predicted SST (One-Step-Ahead)', linestyle='--', color='red', alpha=0.9)
    ax1.set_title(f'SST Forecast: One-Step-Ahead (Teacher Forcing)\nMSE: {mse_onestep:.4f}', fontsize=16)
    ax1.set_xlabel('Date', fontsize=14)
    ax1.set_ylabel('Spatially-Averaged SST (°C)', fontsize=14)
    ax1.legend(loc='upper left')
    ax1.grid(True)
    fig1.tight_layout()
    save_path1 = 'forecast_comparison_onestep.png'
    plt.savefig(save_path1, dpi=300)
    print(f"  Plot 1 saved as {save_path1}")

    # Plot 2: Autoregressive Forecast
    print("Creating Plot 2: Autoregressive Forecast...")
    fig2, ax2 = plt.subplots(figsize=(20, 8))
    ax2.plot(time_axis, actual_data_avg, label='Actual SST', color='blue', linewidth=2)
    ax2.plot(time_axis, autoregressive_preds_denorm, label='Predicted SST (Autoregressive)', linestyle='--', color='purple', alpha=0.9)
    ax2.set_title(f'SST Forecast: Autoregressive (Multi-Step)\nMSE: {mse_autoregressive:.4f}', fontsize=16)
    ax2.set_xlabel('Date', fontsize=14)
    ax2.set_ylabel('Spatially-Averaged SST (°C)', fontsize=14)
    ax2.legend(loc='upper left')
    ax2.grid(True)
    fig2.tight_layout()
    save_path2 = 'forecast_comparison_autoregressive.png'
    plt.savefig(save_path2, dpi=300)
    print(f"  Plot 2 saved as {save_path2}")

if __name__ == '__main__':
    main()

