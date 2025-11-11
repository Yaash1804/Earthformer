# ==============================================================================
# SCRIPT FOR TRAINING THE STUDENT (ConvLSTM) MODEL
#
# (Version 4 - Fixes NoneType error ONLY. Paths are unchanged.)
#
# ==============================================================================

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import logging
import xarray as xr
import sys
import warnings

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
    
    # This is the *new* student model, imported from the path you provided
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
    # +1 is correct because range(N) stops at N-1.
    for i in range(len(data) - total_seq_len + 1):
        sequences.append(data[i:i + total_seq_len])
    if not sequences:
        return None, None
    
    # (Num_seqs, T, C, H, W)
    sequences = np.array(sequences, dtype=np.float32)
    
    # Split into input and target
    x = torch.from_numpy(sequences[:, :in_len])
    y = torch.from_numpy(sequences[:, in_len:])
    return x, y

def get_args_parser():
    """Parses command line arguments for student training."""
    parser = argparse.ArgumentParser(description='Teacher-Student Training for SST Domain Adaptation')

    # --- Model Paths ---
    parser.add_argument('--teacher_ckpt_path', type=str, required=True,
                        help='Path to the pre-trained Earthformer (Teacher) .ckpt file.')
    parser.add_argument('--cfg', type=str, required=True,
                        help='Path to the .yaml config file used to train the teacher (e.g., sst.yaml).')
    parser.add_argument('--student_save_dir', type=str, default='checkpoints/student',
                        help='Directory to save the trained ConvLSTM (Student) model.')

    # --- Data Path (Single .nc file) ---
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the single .nc file (e.g., sst.mon.mean.nc).')
    
    # --- Coordinate Arguments ---
    # Default Base Patch (from your plot script):
    parser.add_argument('--base_lat_slice', type=slice_type, default="15.625:20.625",
                        help='Latitude slice for Base Patch (e.g., "15.625:20.625")')
    parser.add_argument('--base_lon_slice', type=slice_type, default="65.625:72.375",
                        help='Longitude slice for Base Patch (e.g., "65.625:72.375")')
    
    # Student Patch coordinates (USER MUST PROVIDE)
    parser.add_argument('--student_lat_slice', type=slice_type, required=True,
                        help='Latitude slice for Student Patch (e.g., "15.625:20.625")')
    parser.add_argument('--student_lon_slice', type=slice_type, required=True,
                        help='Longitude slice for Student Patch (e.g., "72.375:79.125")')

    # --- Training Parameters ---
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs to train the student model.')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for student training.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for the student model optimizer.')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training (cuda or cpu).')
    
    # DataModule split args (from your plot script)
    parser.add_argument('--train_end_year', type=int, default=2015, help='Last year of training data')
    parser.add_argument('--val_end_year', type=int, default=2020, help='Last year of validation data')

    return parser

def load_and_prep_data(args, hparams):
    """
    Loads and prepares data as seen in the plot_multiple_patches.py script.
    Returns train_loader and val_loader for the student.
    """
    logging.info(f"Loading full dataset from: {args.data_path}")
    ds_full = xr.open_dataset(args.data_path)

    # --- 1. Get Base Patch Stats ---
    logging.info("Calculating normalization stats from Base Patch...")
    base_train_slice = slice(None, str(args.train_end_year))
    train_data_for_stats = ds_full['sst'].sel(
        time=base_train_slice, lat=args.base_lat_slice, lon=args.base_lon_slice
    ).values.astype(np.float32)
    
    mean_base = np.nanmean(train_data_for_stats)
    std_base = np.nanstd(train_data_for_stats)
    logging.info(f"Base Patch Stats: Mean={mean_base:.4f}, Std={std_base:.4f}")

    # --- 2. Get and Normalize Student Patch Data ---
    logging.info("Processing Student Patch data...")
    ds_student = ds_full.sel(lat=args.student_lat_slice, lon=args.student_lon_slice)
    
    student_data_raw = ds_student['sst'].values.astype(np.float32)
    student_time_index = ds_student.get_index("time")
    
    student_data_filled = np.nan_to_num(student_data_raw, nan=mean_base)
    student_data_norm = (student_data_filled - mean_base) / std_base
    student_data_norm = student_data_norm[:, np.newaxis, :, :] # Add channel dim (T, C, H, W)

    # --- 3. Create Train/Val/Test Splits ---
    val_slice = slice(str(args.train_end_year + 1), str(args.val_end_year))
    
    train_indices = student_time_index.slice_indexer(base_train_slice.start, base_train_slice.stop)
    val_indices = student_time_index.slice_indexer(val_slice.start, val_slice.stop)
    
    train_array = student_data_norm[train_indices]
    val_array = student_data_norm[val_indices]
    
    logging.info(f"Student data shapes: Train={train_array.shape}, Val={val_array.shape}")

    # --- 4. Create Datasets and DataLoaders ---
    # hparams is a Read-only Dict, so we access .dataset
    in_len = hparams.dataset.in_len
    out_len = hparams.dataset.out_len
    
    train_x, train_y = create_sequences(train_array, in_len, out_len)
    val_x, val_y = create_sequences(val_array, in_len, out_len)
    
    if train_x is None or val_x is None:
        logging.error("Could not create sequences. Data array might be too short for in_len+out_len.")
        exit(1)
        
    logging.info(f"Created {len(train_x)} training sequences and {len(val_x)} validation sequences.")
    
    train_dataset = TensorDataset(train_x, train_y)
    val_dataset = TensorDataset(val_x, val_y)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
                            
    ds_full.close()
    return train_loader, val_loader

def train_one_epoch(teacher_model, student_model, dataloader, optimizer, criterion, device):
    """Runs a single training epoch based on your specified logic."""
    student_model.train()
    teacher_model.eval()
    
    total_loss = 0
    progress_bar = tqdm(dataloader, desc='Training Epoch', leave=False)

    for input_seq, target_seq in progress_bar:
        # input_seq: (B, 12, 1, H, W)
        # target_seq: (B, 12, 1, H, W)
        input_seq = input_seq.to(device)
        target_seq = target_seq.to(device)
        
        # --- 1. Permute for Teacher ---
        # Teacher expects (B, T, H, W, C)
        input_seq_teacher = input_seq.permute(0, 1, 3, 4, 2)

        # --- 2. Teacher Forward Pass (Frozen) ---
        with torch.no_grad():
            teacher_output_permuted = teacher_model(input_seq_teacher) # (B, 12, H, W, 1)
        
        # --- 3. Permute Back for Student ---
        # Student expects (B, T, C, H, W)
        teacher_output = teacher_output_permuted.permute(0, 1, 4, 2, 3) # (B, 12, 1, H, W)
        
        # --- 4. Get 1st Step (Student Input) ---
        student_input = teacher_output[:, 0:1, :, :, :] # (B, 1, 1, H, W)

        # --- 5. Get 1st Step (Student Target) ---
        student_target = target_seq[:, 0:1, :, :, :] # (B, 1, 1, H, W)

        # --- 6. Student Forward Pass (Trainable) ---
        optimizer.zero_grad()
        student_prediction = student_model(student_input) # (B, 1, 1, H, W)
        
        # --- 7. Loss and Backprop ---
        loss = criterion(student_prediction, student_target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    return avg_loss

def validate_one_epoch(teacher_model, student_model, dataloader, criterion, device):
    """Runs a single validation epoch."""
    student_model.eval()
    teacher_model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for input_seq, target_seq in dataloader:
            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)

            # --- Identical logic to training ---
            input_seq_teacher = input_seq.permute(0, 1, 3, 4, 2)
            teacher_output_permuted = teacher_model(input_seq_teacher)
            teacher_output = teacher_output_permuted.permute(0, 1, 4, 2, 3)
            
            student_input = teacher_output[:, 0:1, :, :, :]
            student_target = target_seq[:, 0:1, :, :, :]
            
            student_prediction = student_model(student_input)
            
            loss = criterion(student_prediction, student_target)
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else 0
    return avg_loss

def main(args):
    """Main function to run the teacher-student training."""
    
    # Clean up special characters from pasted code
    args.student_save_dir = args.student_save_dir.strip()
    
    device = torch.device(args.device)
    logging.info(f"Using device: {device}")

    # --- 1. Load & Freeze Teacher Model (Earthformer) ---
    logging.info(f"Loading Teacher model from checkpoint: {args.teacher_ckpt_path}")
    try:
        # --- THIS IS THE FIX for the 'NoneType' error ---
        # We must pass `save_dir` just like the plotting script does.
        # We can re-use the student_save_dir as a valid path.
        pl_module = CuboidSSTPLModule.load_from_checkpoint(
            args.teacher_ckpt_path,
            oc_file=args.cfg,
            save_dir=args.student_save_dir, # <--- THIS FIXES THE ERROR
            map_location=device
        )
        # ------------------------------------------------
        
        teacher_model = pl_module.torch_nn_module
        hparams = pl_module.hparams # Get hparams from the loaded model
    except Exception as e:
        logging.error(f"Error loading teacher checkpoint: {e}")
        logging.error("Please ensure --cfg points to the correct sst.yaml file.")
        exit(1)
        
    teacher_model.to(device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    logging.info(f"Teacher model (CuboidTransformerModel) loaded and frozen.")

    # --- 2. Setup Data ---
    train_loader, val_loader = load_and_prep_data(args, hparams)
    
    logging.info(f"DataLoaders created. Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # --- 3. Initialize Student Model (ConvLSTM) ---
    student_model = ConvLSTMStudent(
        input_dim=1,
        hidden_dim=12,
        kernel_size=(3, 3),
        num_layers=2
    ).to(device)
    logging.info("ConvLSTM Student model initialized.")

    # --- 4. Setup Optimizer and Criterion ---
    optimizer = optim.Adam(student_model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # --- 5. Training Loop ---
    best_val_loss = float('inf')
    os.makedirs(args.student_save_dir, exist_ok=True)
    
    logging.info("--- Starting Student Training ---")
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(teacher_model, student_model, train_loader, optimizer, criterion, device)
        val_loss = validate_one_epoch(teacher_model, student_model, val_loader, criterion, device)
        
        logging.info(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(args.student_save_dir, 'best_student_model.pth')
            torch.save(student_model.state_dict(), save_path)
            logging.info(f"New best model saved to {save_path}")

    logging.info("--- Student Training Finished ---")
    logging.info(f"Best validation loss: {best_val_loss:.6f}")

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
