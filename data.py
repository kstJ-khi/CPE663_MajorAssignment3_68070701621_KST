"""
data.py — loads the CSV files into PyTorch
"""

import csv
import torch
from torch.utils.data import Dataset, DataLoader

PAD        = 0
VOCAB_SIZE = 5
MAX_LEN    = 20


def load_csv(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            tokens = [int(row[f"token_{i:02d}"]) for i in range(1, MAX_LEN + 1)]
            mask   = [int(row[f"mask_{i:02d}"])  for i in range(1, MAX_LEN + 1)]
            label  = int(row["label"])
            rows.append((tokens, mask, label))
    return rows


class SeqDataset(Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        tokens, mask, label = self.rows[i]
        ids      = torch.tensor(tokens, dtype=torch.long)
        pad_mask = torch.tensor(mask,   dtype=torch.bool).logical_not()
        label    = torch.tensor(label,  dtype=torch.long)
        return ids, pad_mask, label


def get_dataloaders(train_path, val_path, test_path, batch_size=32):
    train = DataLoader(SeqDataset(load_csv(train_path)), batch_size=batch_size, shuffle=True)
    val   = DataLoader(SeqDataset(load_csv(val_path)),   batch_size=batch_size)
    test  = DataLoader(SeqDataset(load_csv(test_path)),  batch_size=batch_size)
    return train, val, test
