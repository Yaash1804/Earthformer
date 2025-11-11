# ==============================================================================
# SCRIPT FOR EVALUATING THE TRAINED (TEACHER + STUDENT) MODEL
#
# (Version 2 - Includes Y-Axis Tick Formatting)
#
# ==============================================================================

# --- ROBUST FIX for Matplotlib Backend Error ---
import matplotlib
matplotlib.use('Agg')
# --- END OF FIX ---

import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import logging
import xarray as xr
import sys
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# --- Add Repository Root to Python Path ---
# This finds the 'src' folder by going up 3 levels
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, '..', '..', '..'))
if _ROOT_DIR not in sys.path:
    sys.path.append(_ROOT_DIR)

# --- Imports from your Earthformer repository ---
try:
    # This is your *original* training script file, used to load the model
    from scripts.cuboid_transformer.sst.train_cuboid_sst import CuboidSSTPLModule
    # This is the *new* student model
    from scripts.student_model import ConvLSTMStudent
except ImportError as e:
    print(f"Error: Could not import modules. {e}")
    print(f"Current Python path: {sys.path}")
    print("Please ensure the file locations are correct.")
    exit(1)

# --- Setup Logging & Warnings ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
warnings.filterwarnings("ignore", "This DataLoader will create .* worker processes .*")
warnings.filterwarnings("ignore", "torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.")

# --- Helper function for parsing slice arguments ---
def slice_type(x):
    """Parses a 'start:end' string into a slice object."""
    try:
        start, end = map(float, x.split(':'))
        return slice(start, end)
    except ValueError:
        raise argparse.ArgumentTypeError("Slice must be in 'start:end' format (e.g., '15.5:20.5')")

def create_sequences(data, in_len, out_len):
    """
    Helper function to create input/target sequences from a time-series array.
    This creates overlapping sequences with a stride of 1.
    """
    sequences = []
    total_seq_len = in_len + out_len
    for i in range(len(data) - total_seq_len + 1):
        sequences.append(data[i:i + total_seq_len])
    if not sequences:
        return None, None
    
    sequences = np.array(sequences, dtype=np.float32)
    x = torch.from_numpy(sequences[:, :in_len])
    y = torch.from_numpy(sequences[:, in_len:])
    return x, y

def get_args_parser():
    """Parses command line arguments for student evaluation."""
    parser = argparse.ArgumentParser(description='Teacher-Student Evaluation Script')

    # --- Model Paths ---
    parser.add_argument('--teacher_ckpt_path', type=str, required=True,
                        help='Path to the pre-trained Earthformer (Teacher) .ckpt file.')
    parser.add_argument('--student_ckpt_path', type=str, required=True,
                        help='Path to the trained ConvLSTM (Student) .pth file.')
    parser.add_argument('--cfg', type=str, required=True,
                        help='Path to the .yaml config file (e.g., sst.yaml).')
    
    # --- Data Path ---
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the single .nc file (e.g., sst.mon.mean.nc).')
    parser.add_argument('--plot_save_path', type=str, default='final_forecast.png',
                        help='Path to save the final comparison plot.')
    
    # --- Coordinate Arguments ---
    parser.add_argument('--base_lat_slice', type=slice_type, default="15.625:20.625",
                        help='Latitude slice for Base Patch (for stats).')
    parser.add_argument('--base_lon_slice', type=slice_type, default="65.625:72.375",
                        help='Longitude slice for Base Patch (for stats).')
    
    parser.add_argument('--student_lat_slice', type=slice_type, required=True,
                        help='Latitude slice for Student Patch (for evaluation).')
    parser.add_argument('--student_lon_slice', type=slice_type, required=True,
                        help='Longitude slice for Student Patch (for evaluation).')

    # --- Eval Parameters ---
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for evaluation (can be larger than training).')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for evaluation (cuda or cpu).')
    parser.add_argument('--train_end_year', type=int, default=2015,
                        help='Last year of training data (used to get base stats).')

    return parser

def load_and_prep_data_for_eval(args, hparams):
    """
    Loads data for the *entire* time series of the student patch,
    but normalizes it using the stats from the base patch.
    """
    logging.info(f"Loading full dataset from: {args.data_path}")
    ds_full = xr.open_dataset(args.data_path)

    # --- 1. Get Base Patch Stats (from training period) ---
    logging.info("Calculating normalization stats from Base Patch (training period)...")
    base_train_slice = slice(None, str(args.train_end_year))
    train_data_for_stats = ds_full['sst'].sel(
        time=base_train_slice, lat=args.base_lat_slice, lon=args.base_lon_slice
    ).values.astype(np.float32)
    
    mean_base = np.nanmean(train_data_for_stats)
    std_base = np.nanstd(train_data_for_stats)
    logging.info(f"Base Patch Stats: Mean={mean_base:.4f}, Std={std_base:.4f}")

    # --- 2. Get and Normalize Student Patch Data (Full Time Series) ---
    logging.info("Processing Student Patch data (full time series)...")
    ds_student = ds_full.sel(lat=args.student_lat_slice, lon=args.student_lon_slice)
    
    student_data_raw = ds_student['sst'].values.astype(np.float32)
    student_time_index = ds_student.get_index("time")
    
    student_data_filled = np.nan_to_num(student_data_raw, nan=mean_base)
    student_data_norm = (student_data_filled - mean_base) / std_base
    student_data_norm = student_data_norm[:, np.newaxis, :, :] # (T, C, H, W)
    
    logging.info(f"Full student data shape (normalized): {student_data_norm.shape}")

    # --- 3. Create Datasets and DataLoaders for *ENTIRE* period ---
    in_len = hparams.dataset.in_len
    out_len = hparams.dataset.out_len
    
    x_full, y_full = create_sequences(student_data_norm, in_len, out_len)
    
    if x_full is None:
        logging.error("Could not create sequences. Data array might be too short.")
        exit(1)
        
    logging.info(f"Created {len(x_full)} total evaluation sequences.")
    
    full_dataset = TensorDataset(x_full, y_full)
    
    # shuffle=False is critical for plotting in time order
    full_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    
    # Calculate the time axis for the *targets* (the points we are predicting)
    # The first target starts at index `in_len`
    time_axis = student_time_index[in_len : in_len + len(y_full)]
                            
    ds_full.close()
    return full_loader, time_axis, mean_base, std_base

def main(args):
    """Main function to run the evaluation."""
    
    device = torch.device(args.device)
    logging.info(f"Using device: {device}")

    # --- 1. Load & Freeze Teacher Model (Earthformer) ---
    logging.info(f"Loading Teacher model from checkpoint: {args.teacher_ckpt_path}")
    try:
        # Use a dummy save_dir, as it's required by the load function
        dummy_save_dir = os.path.dirname(args.student_ckpt_path)
        pl_module = CuboidSSTPLModule.load_from_checkpoint(
            args.teacher_ckpt_path,
            oc_file=args.cfg,
            save_dir=dummy_save_dir,
            map_location=device
        )
        teacher_model = pl_module.torch_nn_module
        hparams = pl_module.hparams
    except Exception as e:
        logging.error(f"Error loading teacher checkpoint: {e}")
        exit(1)
        
    teacher_model.to(device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    logging.info(f"Teacher model (CuboidTransformerModel) loaded and frozen.")

    # --- 2. Load & Freeze Student Model (ConvLSTM) ---
    logging.info(f"Loading Student model from checkpoint: {args.student_ckpt_path}")
    try:
        student_model = ConvLSTMStudent(
            input_dim=1,
            hidden_dim=12,
            kernel_size=(3, 3),
            num_layers=2
        ).to(device)
        student_model.load_state_dict(torch.load(args.student_ckpt_path, map_location=device))
        student_model.eval()
        for param in student_model.parameters():
            param.requires_grad = False
    except Exception as e:
        logging.info(f"Error loading student checkpoint: {e}")
        exit(1)
    logging.info("ConvLSTM Student model loaded and frozen.")

    # --- 3. Setup Data ---
    # This function loads the *entire* time series for the student patch
    full_loader, time_axis, mean_base, std_base = load_and_prep_data_for_eval(args, hparams)
    
    logging.info(f"DataLoaders created. Total evaluation batches: {len(full_loader)}")

    # --- 4. Run Full Evaluation Loop ---
    all_predictions_norm = []
    all_actuals_norm = []
    
    logging.info("--- Running Evaluation over Full Time Series ---")
    
    with torch.no_grad():
        for input_seq, target_seq in tqdm(full_loader, desc="Evaluating"):
            # input_seq: (B, 12, 1, H, W)
            # target_seq: (B, 12, 1, H, W)
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            
            # --- Teacher Pass ---
            input_seq_teacher = input_seq.permute(0, 1, 3, 4, 2) # (B, T, H, W, C)
            teacher_output_permuted = teacher_model(input_seq_teacher) # (B, 12, H, W, 1)
            teacher_output = teacher_output_permuted.permute(0, 1, 4, 2, 3) # (B, 12, 1, H, W)
            
            # --- Student Pass ---
            student_input = teacher_output[:, 0:1, :, :, :] # (B, 1, 1, H, W)
            student_prediction = student_model(student_input) # (B, 1, 1, H, W)
            
            # Get the corresponding target
            student_target = target_seq[:, 0:1, :, :, :] # (B, 1, 1, H, W)

            # --- Spatially Average ---
            # .mean(dim=(2, 3, 4)) reduces (B, 1, 1, H, W) to (B)
            pred_avg = student_prediction.mean(dim=(2, 3, 4))
            actual_avg = student_target.mean(dim=(2, 3, 4))
            
            all_predictions_norm.extend(pred_avg.cpu().numpy())
            all_actuals_norm.extend(actual_avg.cpu().numpy())

    # --- 5. Process and De-normalize Results ---
    logging.info("Evaluation complete. Processing results...")
    preds_norm = np.array(all_predictions_norm)
    actuals_norm = np.array(all_actuals_norm)
    
    preds_denorm = (preds_norm * std_base) + mean_base
    actuals_denorm = (actuals_norm * std_base) + mean_base
    
    # Ensure time_axis and data align
    if len(time_axis) != len(actuals_denorm):
        logging.warning(f"Time axis length ({len(time_axis)}) and data length ({len(actuals_denorm)}) mismatch. Trimming time axis.")
        time_axis = time_axis[:len(actuals_denorm)]
    
    # --- 6. Calculate Final MSE ---
    final_mse = mean_squared_error(actuals_denorm, preds_denorm)
    logging.info(f"--- FINAL MSE (Full Period): {final_mse:.6f} ---")
    
    # --- 7. Plot Results ---
    logging.info(f"Saving plot to {args.plot_save_path}...")
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(20, 8))

        ax.plot(time_axis, actuals_denorm, label='Actual Spatially-Averaged SST', color='blue', linewidth=2)
        ax.plot(time_axis, preds_denorm, label='Predicted SST (Teacher+Student)', 
                linestyle='--', color='red', alpha=0.9, linewidth=1.5)

        title = (f"Teacher+Student Forecast vs. Actual SST (1-Month Ahead)\n"
                 f"Patch: lat=({args.student_lat_slice.start}:{args.student_lat_slice.stop}), "
                 f"lon=({args.student_lon_slice.start}:{args.student_lon_slice.stop})\n"
                 f"Final MSE (Full Period): {final_mse:.4f}")
        ax.set_title(title, fontsize=16)
        
        # --- NEW CODE BLOCK TO SET Y-TICKS AT 1.0 INTERVALS ---
        # Find the min and max values of all plotted data
        all_data = np.concatenate([actuals_denorm, preds_denorm])
        y_min = np.floor(np.nanmin(all_data)) # Floor of the min
        y_max = np.ceil(np.nanmax(all_data))   # Ceil of the max
        
        # Create new ticks at 1-degree intervals
        # Add +1 to y_max so np.arange includes the top value
        y_ticks = np.arange(y_min, y_max + 1, 1.0) 
        ax.set_yticks(y_ticks)
        # --- END OF NEW CODE BLOCK ---
        
        ax.set_xlabel('Date', fontsize=14)
        ax.set_ylabel('Spatially-Averaged SST (°C)', fontsize=14)
        ax.legend(loc='upper left', fontsize=12)
        ax.grid(True)
        fig.tight_layout()

        plt.savefig(args.plot_save_path, dpi=300)
        logging.info("Plot saved successfully.")
    except Exception as e:
        logging.error(f"Error during plotting: {e}")

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
