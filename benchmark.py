"""
benchmark.py — run all 4 models and compare
Run: python benchmark.py
"""

from train import train
from utils import plot_curves, print_table, save_table


MODELS = [
    dict(label="A", num_heads=1, num_layers=1, use_positional_encoding=True),
    dict(label="B", num_heads=4, num_layers=1, use_positional_encoding=True),
    dict(label="C", num_heads=4, num_layers=1, use_positional_encoding=False),
    dict(label="D", num_heads=4, num_layers=2, use_positional_encoding=True),
]

SHARED = dict(
    train_path="data/train.csv",
    val_path="data/validation.csv",
    test_path="data/test.csv",
    embed_dim=64, ff_dim=128, dropout=0.1,
    epochs=15, batch_size=32, lr=0.001, seed=42,
)

if __name__ == "__main__":
    all_results = []

    for cfg in MODELS:
        label = cfg["label"]
        print(f"\n{'='*40}")
        print(f"Model {label}  PE={cfg['use_positional_encoding']}  heads={cfg['num_heads']}  layers={cfg['num_layers']}")
        print(f"{'='*40}")

        results = train(
            **SHARED,
            num_heads               = cfg["num_heads"],
            num_layers              = cfg["num_layers"],
            use_positional_encoding = cfg["use_positional_encoding"],
            save_path               = f"model_{label}.pt",
        )

        plot_curves(results["history"], label, save_path=f"curve_{label}.png")

        all_results.append({**cfg, **{k: results[k] for k in ["val_acc", "test_acc", "train_time", "params"]},
                            "use_pe": cfg["use_positional_encoding"]})

    print_table(all_results)
    save_table(all_results)
