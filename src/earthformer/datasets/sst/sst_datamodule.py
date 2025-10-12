import os
import torch
import xarray as xr
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class SSTDataset(Dataset):
    """
    MODIFIED: This class now accepts a pre-processed data array
    instead of loading data from a file itself. It's much lighter.
    """
    def __init__(self,
                 data: np.ndarray,
                 in_len: int = 12,
                 out_len: int = 12):
        super().__init__()
        self.data = data
        self.in_len = in_len
        self.out_len = out_len
        self.seq_len = in_len + out_len

    def __len__(self):
        return self.data.shape[0] - self.seq_len + 1

    def __getitem__(self, idx):
        sequence = self.data[idx:idx + self.seq_len]
        input_seq = torch.from_numpy(sequence[:self.in_len])
        target_seq = torch.from_numpy(sequence[self.in_len:self.seq_len])
        return input_seq, target_seq


class SSTDataModule(pl.LightningDataModule):
    """
    MODIFIED: This class now handles all data loading and preprocessing
    in the setup() method to avoid redundant operations.
    """
    def __init__(self,
                 data_root: str,
                 in_len: int = 12,
                 out_len: int = 12,
                 batch_size: int = 8,
                 num_workers: int = 4,
                 train_end_year: int = 2015,
                 val_end_year: int = 2020):
        super().__init__()
        print("🐛 [DEBUG] Initializing SSTDataModule...")
        # Save hyperparameters
        self.save_hyperparameters()

        self.sst_train = None
        self.sst_val = None
        self.sst_test = None

    def prepare_data(self):
        # This is for downloading, can be left empty
        pass

    def setup(self, stage: str = None):
        print(f"🚀 [DEBUG] SSTDataModule setup() called for stage: '{stage}'")
        
        # --- Step 1: Load data ONCE ---
        file_path = os.path.join(self.hparams.data_root, "sst.mon.mean.nc")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found at {file_path}")
        
        print(f"⏳ [DEBUG] Loading NetCDF file ONCE from: {file_path}")
        ds = xr.open_dataset(file_path)

        # --- NEW: CROP TO YOUR REGION OF INTEREST ---
        print("🌍 [DEBUG] Cropping dataset to specified lat/lon patch...")
        lat_slice = slice(15.625, 20.625)
        lon_slice = slice(65.625, 72.375)
        ds = ds.sel(lat=lat_slice, lon=lon_slice)
        print(f"✅ [DEBUG] Cropping complete. New dimensions: lat={len(ds.lat)}, lon={len(ds.lon)}")
        # --- END OF NEW STEP ---

        # --- Step 2: Calculate normalization stats correctly ONCE ---
        # Note: Stats are now calculated only on your region of interest
        train_slice = slice(None, str(self.hparams.train_end_year))
        print("⏳ [DEBUG] Calculating normalization stats from the cropped training slice...")
        train_data_for_stats = ds['sst'].sel(time=train_slice).values.astype(np.float32)
        
        # FIX: Use nanmean and nanstd to handle missing values
        mean = np.nanmean(train_data_for_stats)
        std = np.nanstd(train_data_for_stats)
        print(f"📊 [DEBUG] Normalization stats for patch: mean={mean:.4f}, std={std:.4f}")

        # --- Step 3: Preprocess full data ONCE ---
        print("⏳ [DEBUG] Processing cropped dataset...")
        full_data = ds['sst'].values.astype(np.float32)
        
        # FIX: Fill NaNs using the correctly calculated mean
        full_data = np.nan_to_num(full_data, nan=mean)
        
        # Normalize the entire dataset
        full_data_normalized = (full_data - mean) / std
        # Add channel dimension
        full_data_normalized = full_data_normalized[:, np.newaxis, :, :]
        print(f"✅ [DEBUG] Cropped dataset processed. Shape: {full_data_normalized.shape}")

        # --- Step 4: Split data into arrays ---
        time_index = ds.get_index("time")
        val_slice = slice(str(self.hparams.train_end_year + 1), str(self.hparams.val_end_year))
        test_slice = slice(str(self.hparams.val_end_year + 1), None)

        train_indices = time_index.slice_indexer(train_slice.start, train_slice.stop)
        val_indices = time_index.slice_indexer(val_slice.start, val_slice.stop)
        test_indices = time_index.slice_indexer(test_slice.start, test_slice.stop)
        
        train_array = full_data_normalized[train_indices]
        val_array = full_data_normalized[val_indices]
        test_array = full_data_normalized[test_indices]
        
        print(f"📊 [DEBUG] Train array shape: {train_array.shape}")
        print(f"📊 [DEBUG] Val array shape:   {val_array.shape}")
        print(f"📊 [DEBUG] Test array shape:  {test_array.shape}")

        # --- Step 5: Instantiate lightweight datasets ---
        if stage == "fit" or stage is None:
            self.sst_train = SSTDataset(data=train_array, in_len=self.hparams.in_len, out_len=self.hparams.out_len)
            self.sst_val = SSTDataset(data=val_array, in_len=self.hparams.in_len, out_len=self.hparams.out_len)
        if stage == "test" or stage is None:
            self.sst_test = SSTDataset(data=test_array, in_len=self.hparams.in_len, out_len=self.hparams.out_len)
        
        print("✅ [DEBUG] SSTDataModule setup() finished.")

    def train_dataloader(self):
        return DataLoader(self.sst_train, batch_size=self.hparams.batch_size, shuffle=True,
                          num_workers=self.hparams.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.sst_val, batch_size=self.hparams.batch_size, shuffle=False,
                          num_workers=self.hparams.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.sst_test, batch_size=self.hparams.batch_size, shuffle=False,
                          num_workers=self.hparams.num_workers, pin_memory=True)
