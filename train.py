"""
train.py — train one model variant, return results
"""

import time
import torch
import torch.nn as nn
from torch.optim import Adam

from data  import get_dataloaders
from model import MiniTransformer


def run_epoch(model, loader, optimizer, loss_fn, device, train=True):
    model.train() if train else model.eval()
    loss_sum, correct, total = 0, 0, 0

    with torch.set_grad_enabled(train):
        for ids, mask, labels in loader:
            ids, mask, labels = ids.to(device), mask.to(device), labels.to(device)
            logits = model(ids, mask)
            loss   = loss_fn(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            correct  += (logits.argmax(1) == labels).sum().item()
            loss_sum += loss.item() * len(labels)
            total    += len(labels)

    return loss_sum / total, correct / total


def train(
    train_path="data/train.csv", val_path="data/validation.csv", test_path="data/test.csv",
    vocab_size=5, embed_dim=64, num_heads=4, ff_dim=128, num_layers=1,
    dropout=0.1, use_positional_encoding=True,
    epochs=15, batch_size=32, lr=0.001, seed=42, save_path="best_model.pt",
):
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = get_dataloaders(train_path, val_path, test_path, batch_size)

    model     = MiniTransformer(vocab_size, embed_dim, num_heads, ff_dim, num_layers,
                                dropout, use_positional_encoding).to(device)
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    print(f"  params: {model.count_parameters():,} | device: {device}\n")

    history      = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    start        = time.time()

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer, loss_fn, device, train=True)
        vl_loss, vl_acc = run_epoch(model, val_loader,   optimizer, loss_fn, device, train=False)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)

        print(f"  epoch {epoch:02d}/{epochs}  train {tr_acc:.3f}  val {vl_acc:.3f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), save_path)

    train_time = (time.time() - start) / 60

    model.load_state_dict(torch.load(save_path, map_location=device))
    _, test_acc = run_epoch(model, test_loader, optimizer, loss_fn, device, train=False)

    print(f"\n  done — val {best_val_acc:.3f}  test {test_acc:.3f}  time {train_time:.1f}min\n")

    return {
        "history":    history,
        "val_acc":    round(best_val_acc, 4),
        "test_acc":   round(test_acc, 4),
        "train_time": round(train_time, 2),
        "params":     model.count_parameters(),
    }


if __name__ == "__main__":
    train(num_heads=4, num_layers=1, use_positional_encoding=True, save_path="model_B.pt")
