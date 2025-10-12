import os
import torch
import xarray as xr
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class SSTDataset(Dataset):
    """
    PyTorch Dataset for loading a single, continuous monthly SST NetCDF file.
    """
    def __init__(self,
                 data_root: str,
                 split: str = "train",
                 in_len: int = 12,
                 out_len: int = 12,
                 train_end_year: int = 2015,
                 val_end_year: int = 2020):
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.in_len = in_len
        self.out_len = out_len
        self.seq_len = in_len + out_len

        # --- Load NetCDF file ---
        file_path = os.path.join(self.data_root, "sst.mon.mean.nc")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found at {file_path}")
        
        ds = xr.open_dataset(file_path)
        full_data = ds['sst'].values.astype(np.float32)

        # Handle NaNs
        if np.isnan(full_data).any():
            global_mean = np.nanmean(full_data)
            full_data = np.nan_to_num(full_data, nan=global_mean)

        # --- Split based on datetime coordinates ---
        train_slice = slice(None, str(train_end_year))
        val_slice = slice(str(train_end_year + 1), str(val_end_year))
        test_slice = slice(str(val_end_year + 1), None)

        # Normalize using training data
        train_data = ds['sst'].sel(time=train_slice).values.astype(np.float32)
        self.mean = np.mean(train_data)
        self.std = np.std(train_data)
        full_data_normalized = (full_data - self.mean) / self.std

        # Select split
        time_index = ds.get_index("time")
        if self.split == "train":
            indices = time_index.slice_indexer(train_slice.start, train_slice.stop)
        elif self.split == "val":
            indices = time_index.slice_indexer(val_slice.start, val_slice.stop)
        elif self.split == "test":
            indices = time_index.slice_indexer(test_slice.start, test_slice.stop)
        else:
            raise ValueError(f"Invalid split '{self.split}'. Must be 'train', 'val', or 'test'.")

        self.data = full_data_normalized[indices]
        self.data = self.data[:, np.newaxis, :, :]  # Add channel dim

    def __len__(self):
        return self.data.shape[0] - self.seq_len + 1

    def __getitem__(self, idx):
        sequence = self.data[idx:idx + self.seq_len]
        input_seq = torch.from_numpy(sequence[:self.in_len])
        target_seq = torch.from_numpy(sequence[self.in_len:self.seq_len])
        return input_seq, target_seq


class SSTDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for SST dataset.
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
        self.data_root = data_root
        self.in_len = in_len
        self.out_len = out_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_end_year = train_end_year
        self.val_end_year = val_end_year

        self.sst_train = None
        self.sst_val = None
        self.sst_test = None

    def prepare_data(self):
        pass

    def setup(self, stage: str = None):
        """Instantiate datasets for each split."""
        if stage == "fit" or stage is None:
            self.sst_train = SSTDataset(
                data_root=self.data_root,
                split="train",
                in_len=self.in_len,
                out_len=self.out_len,
                train_end_year=self.train_end_year,
                val_end_year=self.val_end_year
            )
            self.sst_val = SSTDataset(
                data_root=self.data_root,
                split="val",
                in_len=self.in_len,
                out_len=self.out_len,
                train_end_year=self.train_end_year,
                val_end_year=self.val_end_year
            )
        if stage == "test" or stage is None:
            self.sst_test = SSTDataset(
                data_root=self.data_root,
                split="test",
                in_len=self.in_len,
                out_len=self.out_len,
                train_end_year=self.train_end_year,
                val_end_year=self.val_end_year
            )

    def train_dataloader(self):
        return DataLoader(self.sst_train, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.sst_val, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.sst_test, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)
