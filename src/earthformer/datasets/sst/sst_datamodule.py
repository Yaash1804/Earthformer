import os
import torch
import xarray as xr
import numpy as np
from torch.utils.data import Dataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader

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
        """
        Args:
            data_root (str): Path to the directory containing 'sst.mon.mean.nc'.
            split (str): One of "train", "val", or "test".
            in_len (int): Number of months in the input sequence.
            out_len (int): Number of months in the output/target sequence.
            train_end_year (int): The last year of the training set.
            val_end_year (int): The last year of the validation set.
        """
        super().__init__()
        self.data_root = data_root
        self.split = split
        self.in_len = in_len
        self.out_len = out_len
        self.seq_len = in_len + out_len

        # --- MODIFICATION START: Load a single NetCDF file ---
        file_path = os.path.join(self.data_root, "sst.mon.mean.nc")
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found at {file_path}")
        
        ds = xr.open_dataset(file_path)
        # --- MODIFICATION END ---
        
        full_data = ds['sst'].values.astype(np.float32)
        
        if np.isnan(full_data).any():
            global_mean = np.nanmean(full_data)
            full_data = np.nan_to_num(full_data, nan=global_mean)

        # --- MODIFICATION START: Split data based on datetime coordinates ---
        # Define date ranges for splitting the data
        train_slice = slice(None, str(train_end_year))
        val_slice = slice(str(train_end_year + 1), str(val_end_year))
        test_slice = slice(str(val_end_year + 1), None)

        # Calculate normalization stats ONLY from the training set
        train_data = ds['sst'].sel(time=train_slice).values.astype(np.float32)
        self.mean = np.mean(train_data)
        self.std = np.std(train_data)

        # Normalize the entire dataset
        full_data_normalized = (full_data - self.mean) / self.std

        # Select the correct data split based on the dates
        if self.split == "train":
            self.data = full_data_normalized[ds.get_index("time").slice_indexer(train_slice.start, train_slice.stop)]
        elif self.split == "val":
            self.data = full_data_normalized[ds.get_index("time").slice_indexer(val_slice.start, val_slice.stop)]
        elif self.split == "test":
            self.data = full_data_normalized[ds.get_index("time").slice_indexer(test_slice.start, test_slice.stop)]
        else:
            raise ValueError(f"Invalid split '{self.split}'. Must be 'train', 'val', or 'test'.")
        # --- MODIFICATION END ---
            
        # Add a channel dimension: (T, H, W) -> (T, C, H, W) where C=1
        self.data = self.data[:, np.newaxis, :, :]

    def __len__(self):
        """Returns the total number of possible sequences in the dataset split."""
        return self.data.shape[0] - self.seq_len + 1

    def __getitem__(self, idx):
        """
        Returns a single sample of (input_sequence, target_sequence).
        """
        sequence = self.data[idx:idx + self.seq_len]
        input_seq = torch.from_numpy(sequence[:self.in_len])
        target_seq = torch.from_numpy(sequence[self.in_len:self.seq_len])
        return input_seq, target_seq

class SSTDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for the custom SST dataset.
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
        self.save_hyperparameters()
        self.sst_train = None
        self.sst_val = None
        self.sst_test = None

    def prepare_data(self):
        pass

    def setup(self, stage: str = None):
        """Instantiates the Dataset objects for each split."""
        if stage == "fit" or stage is None:
            self.sst_train = SSTDataset(split="train", **self.hparams)
            self.sst_val = SSTDataset(split="val", **self.hparams)
        if stage == "test" or stage is None:
            self.sst_test = SSTDataset(split="test", **self.hparams)

    def train_dataloader(self):
        return DataLoader(self.sst_train, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.sst_val, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.sst_test, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True)