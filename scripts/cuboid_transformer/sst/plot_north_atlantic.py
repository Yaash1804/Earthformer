# ==============================================================================
# SCRIPT FOR PLOTTING PREDICTIONS ON A NEW, SHIFTED PATCH
#
# This version uses the index method to select the data patch, ensuring the
# dimensions match the model's expectation.
#
# INSTRUCTIONS:
# 1. Save this file as `plot_new_patch.py`.
# 2. Update the `CHECKPOINT_PATH` and other configuration variables below.
# 3. Run from a Colab cell using the command:
#    !MPLBACKEND=Agg python /path/to/your/plot_new_patch.py
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

warnings.filterwarnings("ignore", "This DataLoader will create .* worker processes .*")
warnings.filterwarnings("ignore", "torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.")

from train_cuboid_sst import CuboidSSTPLModule

def create_sequences(data, in_len, out_len):
    """Helper function to create input/target sequences from a time-series array."""
    sequences = []
    total_seq_len = in_len + out_len
    for i in range(len(data) - total_seq_len + 1):
        sequences.append(data[i:i + total_seq_len])
    if not sequences:
        return None, None
    sequences = np.array(sequences)
    x = sequences[:, :in_len]
    y = sequences[:, in_len:]
    return torch.from_numpy(x).float(), torch.from_numpy(y).float()

def plot_new_patch_predictions():
    """
    Main function to load model, process new patch data, and plot predictions.
    """
    # --- 1. CONFIGURATION ---
    print("Step 1: Configuring paths and parameters...")

    CFG_PATH = "/content/Earthformer/scripts/cuboid_transformer/sst/sst.yaml"
    DATA_ROOT = "/content/drive/MyDrive/sst"
    CHECKPOINT_PATH = "/content/Earthformer/scripts/cuboid_transformer/sst/experiments/sst_colab_run_1/checkpoints/model-epoch=019.ckpt"
    save_directory_name = "sst_colab_run_1"

    # --- Define the Center of the NEW Patch ---
    # This is centered slightly to the West (left) of the original training patch.
    center_lat, center_lon = 40, 320
    # Define the required grid size (must match the model's trained input shape)
    patch_height, patch_width = 21, 28

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

    # --- 3. LOAD DATA & GET ORIGINAL NORMALIZATION STATS ---
    print("\nStep 3: Loading SST data and calculating original normalization stats...")
    hparams = model.hparams
    file_path = os.path.join(DATA_ROOT, "sst.mon.mean.nc")
    ds_full = xr.open_dataset(file_path)

    print("  Calculating mean/std from the original training data configuration...")
    original_train_end_year = hparams.dataset.get('train_end_year', 2015)
    original_lat_slice = slice(15.625, 20.625)
    original_lon_slice = slice(65.625, 72.375)
    original_train_slice = slice(None, str(original_train_end_year))
    
    train_data_for_stats = ds_full['sst'].sel(
        time=original_train_slice, lat=original_lat_slice, lon=original_lon_slice
    ).values.astype(np.float32)
    
    mean_original = np.nanmean(train_data_for_stats)
    std_original = np.nanstd(train_data_for_stats)
    print(f"  Original Training Mean: {mean_original:.4f}, Original Training Std: {std_original:.4f}")

    # --- 4. PROCESS THE NEW PATCH (CORRECTED INDEX METHOD) ---
    print(f"\nStep 4: Extracting and processing a {patch_height}x{patch_width} patch centered near ({center_lat}N, {center_lon}E)...")
    
    # Find the grid indices closest to the desired center point
    center_lat_idx = np.abs(ds_full.lat.values - center_lat).argmin()
    center_lon_idx = np.abs(ds_full.lon.values - center_lon).argmin()

    # Calculate the start and end indices to carve out the patch
    start_lat_idx = center_lat_idx - patch_height // 2
    end_lat_idx = start_lat_idx + patch_height
    start_lon_idx = center_lon_idx - patch_width // 2
    end_lon_idx = start_lon_idx + patch_width
    
    # Ensure indices are within bounds
    if start_lat_idx < 0 or end_lat_idx > len(ds_full.lat) or start_lon_idx < 0 or end_lon_idx > len(ds_full.lon):
        raise ValueError("The selected patch center is too close to the edge of the dataset. Please choose a more central point.")

    # Select the data using the calculated integer-based indices to guarantee the shape
    ds_new_patch = ds_full.isel(lat=slice(start_lat_idx, end_lat_idx), lon=slice(start_lon_idx, end_lon_idx))
    
    new_patch_data_raw = ds_new_patch['sst'].values.astype(np.float32)
    new_patch_time_index = ds_new_patch.get_index("time")
    
    # Verify the shape to prevent the error
    print(f"  Successfully extracted patch with spatial shape: {new_patch_data_raw.shape[1:]}")
    assert new_patch_data_raw.shape[1:] == (patch_height, patch_width), "Extracted patch has incorrect dimensions!"

    # Normalize the new data using the OLD statistics
    new_patch_data_filled = np.nan_to_num(new_patch_data_raw, nan=mean_original)
    new_patch_normalized = (new_patch_data_filled - mean_original) / std_original
    new_patch_normalized = new_patch_normalized[:, np.newaxis, :, :] # Add channel dimension

    in_len, out_len = hparams.dataset.in_len, hparams.dataset.out_len
    input_seqs, target_seqs = create_sequences(new_patch_normalized, in_len, out_len)
    
    if input_seqs is None:
        print("  Could not create any sequences from the new patch. The time series may be too short.")
        return
        
    print(f"  Created {len(input_seqs)} sequences for the new patch.")

    # --- 5. RUN INFERENCE ---
    print("\nStep 5: Running model inference on the new patch...")
    all_preds_norm = []
    with torch.no_grad():
        batch_size = hparams.dataset.batch_size
        for i in range(0, len(input_seqs), batch_size):
            print(f"  Processing batch {i//batch_size + 1}/{ -(-len(input_seqs)//batch_size)}...", end='\r')
            x_batch = input_seqs[i:i+batch_size].to(DEVICE)
            pred_y_batch = model.torch_nn_module(x_batch.permute(0, 1, 3, 4, 2))
            pred_y_batch = pred_y_batch.permute(0, 1, 4, 2, 3)
            avg_preds = pred_y_batch.mean(dim=[3, 4])
            all_preds_norm.append(avg_preds[:, 0].cpu().numpy())
    
    all_preds_norm = np.concatenate(all_preds_norm)
    print("\nInference complete.")

    # --- 6. DE-NORMALIZE & PREPARE FOR PLOTTING ---
    print("\nStep 6: De-normalizing data and preparing for plotting...")
    all_preds_denorm = (all_preds_norm * std_original) + mean_original
    actual_data_norm = target_seqs.mean(dim=[3, 4])[:, 0].cpu().numpy()
    actual_data_denorm = (actual_data_norm * std_original) + mean_original
    time_axis = new_patch_time_index[in_len : len(actual_data_denorm) + in_len]

    # --- 7. CREATE THE PLOT ---
    print("\nStep 7: Creating the plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(20, 8))

    ax.plot(time_axis, actual_data_denorm, label='Actual SST (New Patch)', color='blue', linewidth=2)
    ax.plot(time_axis, all_preds_denorm, label='Predicted SST (1-Month Ahead)', linestyle='--', color='red', alpha=0.9)

    ax.set_title(f'SST Forecast Generalization: Shifted Indian Ocean Patch ({center_lat}°N, {center_lon}°E)', fontsize=18)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('Spatially-Averaged SST (°C)', fontsize=14)
    ax.legend(loc='upper left', fontsize=12)
    ax.grid(True)
    fig.tight_layout()

    save_path = 'sst_left_shifted_indian_ocean_predictions.png'
    plt.savefig(save_path, dpi=300)
    print(f"\nPlot saved successfully as {save_path}")

if __name__ == '__main__':
    plot_new_patch_predictions()

