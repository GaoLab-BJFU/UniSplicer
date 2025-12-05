import h5py
import numpy as np
import torch
import torch.optim as optim
import time
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import os
import math
from math import ceil
from Bio import SeqIO
import sys
import random
from torch.utils.data import Subset, DataLoader, Dataset
import argparse
import logging
import torch.optim.lr_scheduler as lr_scheduler

########################################################
#                 Example training code
########################################################

# Training from scratch:
# python Step3_UniSplicer_model_training_source_code.py --species Arabidopsis_thaliana --batchsize 32 --cnn_hidden_unit 60 --lstm_hidden_unit 60 --lstm_layer_num 3 --window_context 600 --epoch_number 10 --lr_rate 1e-3 --lossweight 10.0&

# Or prefer nohup:

# nohup python Step3_UniSplicer_model_training_source_code.py --species Arabidopsis_thaliana --batchsize 32 --cnn_hidden_unit 60 --lstm_hidden_unit 60 --lstm_layer_num 3 --window_context 600 --epoch_number 10 --lr_rate 1e-3 --lossweight 10.0&

# Transfer learning:
# make sure you have the base UniSplicer models first, and then add the transfer learning flag:"--enable_transfer_learning"

########################################################
#                 Reproducibility Setup
########################################################


def set_seed(seed=42, deterministic=True):
    """
    Seed every RNG (python, numpy, torch) and optionally
    force deterministic CUDA kernels for exact reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


set_seed(42, deterministic=True)


########################################################
#                Custom HDF5 Dataset
########################################################
class HDF5SpliceDataset(Dataset):
    """
    Treat an HDF5 training file produced earlier as a PyTorch Dataset.
    Each sample corresponds to one sliding window (seq & labels).
    """

    def __init__(self, h5_path):
        super().__init__()
        self.h5_path = h5_path
        self.h5_file = h5py.File(self.h5_path, 'r')

        # Build index → (x_key, y_key, row_idx) mapping
        self.index_map = []
        self.total_samples = 0
        x_keys = sorted(
            [k for k in self.h5_file.keys() if k.startswith('I')],
            key=lambda x: int(x[1:])
        )
        for x_key in x_keys:
            chunk_index = x_key[1:]
            y_key = 'L' + chunk_index
            X_dataset = self.h5_file[x_key]
            N = X_dataset.shape[0]
            for row_idx in range(N):
                self.index_map.append((x_key, y_key, row_idx))
            self.total_samples += N

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        """
        Return a single window as (X, Y) tensors.
        - X: [4, win_len] float32
        - Y: [win_len, 3] float32
        """
        x_key, y_key, row_idx = self.index_map[idx]
        X_np = self.h5_file[x_key][row_idx]
        Y_np = self.h5_file[y_key][0, row_idx]
        X_t = torch.from_numpy(X_np).float().permute(1, 0)
        Y_t = torch.from_numpy(Y_np).float()
        return X_t, Y_t

    def close(self):
        """Close the underlying HDF5 handle (call when done)."""
        self.h5_file.close()


########################################################
#              Split + DataLoader creation
########################################################
def split_dataset(dataset, train_ratio=0.9, seed=42):
    """Randomly split dataset indices into train / val subsets."""
    total_len = len(dataset)
    indices = list(range(total_len))
    random.seed(seed)
    random.shuffle(indices)
    train_size = int(train_ratio * total_len)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def make_dataloaders(h5_path, batch_size=8, train_ratio=0.9, seed=42):
    """Return PyTorch DataLoaders for training and validation."""
    ds = HDF5SpliceDataset(h5_path)
    train_ds, val_ds = split_dataset(ds, train_ratio=train_ratio, seed=seed)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        drop_last=False, num_workers=0, generator=generator
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        drop_last=False, num_workers=0, generator=generator
    )
    return train_loader, val_loader, ds


########################################################
#              Model Definitions
########################################################
class SelfAttention(nn.Module):
    """Single-head self-attention over sequence length dimension."""

    def __init__(self, hidden_unit: int):
        super().__init__()
        self.hidden_unit = hidden_unit
        self.query = nn.Linear(hidden_unit, hidden_unit)
        self.key = nn.Linear(hidden_unit, hidden_unit)
        self.value = nn.Linear(hidden_unit, hidden_unit)

    def forward(self, x: torch.Tensor):
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        attention_weights = F.softmax(
            Q @ K.transpose(-2, -1) / (self.hidden_unit ** 0.5), dim=-1
        )
        return attention_weights @ V


class ResidualCNNBlock(nn.Module):
    """Two conv-BN-ReLU layers with residual connection."""

    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.block(x))


class CNN_LSTM_Attention(nn.Module):
    """
    Input  : [B, 4, L]  (one-hot seq)
    Output : [B, 3, L-crop]  (class scores per base)
    """

    def __init__(self, input_shape, hidden_unit, lstm_hidden_unit, num_layers,
                 crop_size, num_classes):
        super().__init__()
        self.crop_size = crop_size

        # --- Convolutional front-end ------------------------------------
        self.conv_block_1 = nn.Sequential(
            nn.Conv1d(input_shape, hidden_unit, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_unit),
            nn.ReLU(),
            nn.Conv1d(hidden_unit, hidden_unit, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_unit),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv1d(hidden_unit, hidden_unit, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_unit),
            nn.ReLU(),
            nn.Conv1d(hidden_unit, hidden_unit, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_unit),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        )
        self.cnn_backbone = nn.Sequential(
            ResidualCNNBlock(hidden_unit),
            ResidualCNNBlock(hidden_unit),
            ResidualCNNBlock(hidden_unit),
        )

        # --- Recurrent + attention heads --------------------------------
        self.lstm = nn.LSTM(
            hidden_unit, lstm_hidden_unit, num_layers,
            batch_first=True, bidirectional=True
        )
        self.attention = SelfAttention(hidden_unit=2 * lstm_hidden_unit)
        self.norm1 = nn.LayerNorm(2 * lstm_hidden_unit)
        self.classifier = nn.Linear(2 * lstm_hidden_unit, num_classes)

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.cnn_backbone(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = self.attention(x)
        x = self.norm1(x)
        x = self.classifier(x)
        x = x.transpose(1, 2)
        # Center-crop to match original window length minus context padding
        left_crop = self.crop_size // 2
        right_crop = self.crop_size - left_crop
        if right_crop > 0:
            x = x[:, :, left_crop:-right_crop]
        else:
            x = x[:, :, left_crop:]
        return x


# ---------------- Utility functions ------------------
def one_hot_to_index(y_oh: torch.Tensor) -> torch.Tensor:
    """
    Convert one-hot or sparse-one-hot labels → integer class indices,
    mapping all-zero rows to 3 (‘padding’).
    """
    B, C, L = y_oh.shape
    sums = y_oh.sum(dim=1)
    maxidx = y_oh.argmax(dim=1)
    pad_mask = (sums == 0)
    maxidx[pad_mask] = 3
    return maxidx


def categorical_crossentropy_2d(y_true, y_pred, eps=1e-10):
    """
    Alternative (unused) 2-D categorical cross-entropy supporting one-hot.
    """
    if y_true.shape[-1] == 3 and y_true.dim() == 3:
        y_true = y_true.permute(0, 2, 1)
    y_pred = F.softmax(y_pred, dim=1)
    crossent = -torch.sum(y_true * torch.log(y_pred + eps), dim=1)
    return crossent.mean()


########################################################
#             Training Function
########################################################
def train_spliceai_with_dataloader(
    species: str,
    model_parameter: nn.Module,
    version: str,
    h5_path: str,
    loss_weight: float,
    run_idx: int,
    window_ctx: int,
    num_epochs: int = 5,
    batch_size: int = 8,
    lr: float = 1e-3,
):
    """
    Complete training routine: dataloaders → model → checkpoints.
    Returns the trained PyTorch model instance.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info("Using device: %s", device)

    # ---------- Data ----------
    train_loader, val_loader, dataset = make_dataloaders(
        h5_path, batch_size=batch_size, train_ratio=0.90, seed=42
    )

    # ---------- Model / Loss / Optim ----------
    model = model_parameter.to(device)
    weights = torch.tensor([1.0, loss_weight, loss_weight], device=device)
    criterion = nn.CrossEntropyLoss(weight=weights, ignore_index=3)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    # ---------- Checkpoint dirs ----------
    training_dir = f"/data/Step3_model_training/{species}"
    result_saving_dir = f"/results/Step3_model_training/{species}"
    run_dir = f"{result_saving_dir}/Models_{version}"
    best_dir = f"{result_saving_dir}/best_models"
    os.makedirs(run_dir,  exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)

    best_ckpt_path = f"{best_dir}/{species}_{window_ctx}_{run_idx}.pt"
    best_val_loss = float("inf")

    start_time = time.time()

    # ---------- Training loop ----------
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)                # [B, 3, L]
            Y_batch = Y_batch.permute(0, 2, 1)      # → [B, 3, L]
            loss = criterion(outputs, one_hot_to_index(Y_batch))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        logging.info("[Epoch %d] Training loss: %.4f",
                     epoch + 1, avg_train_loss)

        # Save epoch checkpoint (always)
        torch.save(model.state_dict(), f"{run_dir}/model_epoch{epoch+1}.pt")

        # ---------- Validation ----------
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_val, Y_val in val_loader:
                X_val = X_val.to(device)
                Y_val = Y_val.to(device)
                preds = model(X_val)
                Y_val = Y_val.permute(0, 2, 1)
                val_loss += criterion(preds, one_hot_to_index(Y_val)).item()
        val_loss /= len(val_loader)
        logging.info("Validation loss: %.4f", val_loss)

        # ---------- Keep the best ----------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_ckpt_path)
            logging.info("  ↳ New best model saved to %s (val_loss = %.4f)",
                         best_ckpt_path, best_val_loss)

        logging.info("Elapsed time: %.1fs", time.time() - start_time)
        scheduler.step()
        logging.info("Learning rate after epoch %d: %f",
                     epoch + 1, scheduler.get_last_lr()[0])

    dataset.close()
    return model


########################################################
#       Logger writer for catching prints/tqdm outputs
########################################################
class LoggerWriter:
    """
    Redirect stdout / stderr into the logging module inside notebooks / CLIs.
    """

    def __init__(self, level):
        self.level = level

    def write(self, message):
        if message and message.strip():
            self.level(message.strip())

    def flush(self):
        pass


########################################################
#               Main Execution
########################################################
if __name__ == "__main__":
    # ------------- CLI argument parsing -------------
    parser = argparse.ArgumentParser(
        description="Train UniSplicer. Required parameters are defined as named arguments."
    )
    parser.add_argument("--species", type=str, required=True,
                        help="Species name (e.g., Arabidopsis_thaliana).")
    parser.add_argument("--batchsize", type=int,
                        required=True, help="Batch size for training.")
    parser.add_argument("--cnn_hidden_unit", type=int, required=True,
                        help="Number of hidden units in CNN layers.")
    parser.add_argument("--lstm_hidden_unit", type=int, required=True,
                        help="Number of hidden units in LSTM layers.")
    parser.add_argument("--lstm_layer_num", type=int,
                        required=True, help="Number of LSTM layers.")
    parser.add_argument("--window_context", type=int, required=True,
                        help="Window context size (also used for cropping).")
    parser.add_argument("--epoch_number", type=int,
                        required=True, help="Number of training epochs.")
    parser.add_argument("--lr_rate", type=float,
                        required=True, help="Learning rate.")
    parser.add_argument("--lossweight", type=float, required=True,
                        help="Loss weight for splice signal classes.")
    parser.add_argument("--enable_transfer_learning", action="store_true",
                        help="Enable transfer learning by loading pretrained base model.")
    args = parser.parse_args()

    # ------------- Repeat training 5× with different seeds -------------
    for i in range(1, 6):
        seed = 41 + i
        set_seed(seed, deterministic=True)

        # Distinguish run directories / filenames
        if args.enable_transfer_learning:
            version = f'transfer_learning_{args.window_context}_{i}'
        else:
            version = f'train_from_beginning_{args.window_context}_{i}'

        base_dir = '/results/Step3_model_training'
        log_file = (f"{base_dir}/{args.species}/Models_{version}/"
                    "model_training.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        # ------------- Configure logging -------------
        logging.basicConfig(
            filename=log_file,
            filemode='w',
            format='%(asctime)s - %(message)s',
            level=logging.INFO
        )
        sys.stdout = LoggerWriter(logging.info)
        sys.stderr = LoggerWriter(logging.error)

        # Log CLI invocation & process ID
        logging.info("Command invoked: %s", ' '.join(sys.argv))
        logging.info("Process ID: %d", os.getpid())

        # ------------- Build model -------------
        model = CNN_LSTM_Attention(
            input_shape=4,
            hidden_unit=args.cnn_hidden_unit,
            lstm_hidden_unit=args.lstm_hidden_unit,
            num_layers=args.lstm_layer_num,
            crop_size=args.window_context,
            num_classes=3,
        )

        # Optional transfer learning from Arabidopsis base model
        if args.enable_transfer_learning:
            base_model_dir = "/results/Step3_model_training/Arabidopsis_thaliana/best_models"
            base_model_file = f"Arabidopsis_thaliana_{args.window_context}_{i}.pt"
            base_model_path = os.path.join(base_model_dir, base_model_file)
            if os.path.exists(base_model_path):
                logging.info("Loading pretrained weights from %s",
                             base_model_path)
                model.load_state_dict(torch.load(
                    base_model_path, map_location='cpu'))
            else:
                logging.warning(
                    "No base model found at %s. Training from scratch.", base_model_path)
        else:
            logging.info("Transfer learning disabled; training from scratch.")

        # ------------- Launch training -------------
        trained_model = train_spliceai_with_dataloader(
            species=args.species,
            model_parameter=model,
            version=version,
            h5_path=(f"/data/Step3_model_training/"
                     f"{args.species}/{args.species}"
                     f"_training_dataset_window_context_{args.window_context}.h5"),
            loss_weight=args.lossweight,
            run_idx=i,
            window_ctx=args.window_context,
            num_epochs=args.epoch_number,
            batch_size=args.batchsize,
            lr=args.lr_rate,
        )
