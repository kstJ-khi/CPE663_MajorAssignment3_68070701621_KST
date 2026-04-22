"""
utils.py — plotting and table helpers
"""

import matplotlib.pyplot as plt


def plot_curves(history, label, save_path=None):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Model {label}")

    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"],   label="val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["val_acc"],   label="val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"  saved plot -> {save_path}")
    plt.show()


def print_table(results):
    print(f"\n{'Model':<8}{'PE':<6}{'Heads':<7}{'Layers':<8}{'Val Acc':<10}{'Test Acc':<10}{'Time(min)':<11}{'Params'}")
    print("-" * 68)
    for r in results:
        print(f"{r['label']:<8}"
              f"{'Yes' if r['use_pe'] else 'No':<6}"
              f"{r['num_heads']:<7}"
              f"{r['num_layers']:<8}"
              f"{r['val_acc']:<10}"
              f"{r['test_acc']:<10}"
              f"{r['train_time']:<11}"
              f"{r['params']:,}")
    print()


def save_table(results, path="benchmark_results.txt"):
    lines = [
        f"{'Model':<8}{'PE':<6}{'Heads':<7}{'Layers':<8}{'Val Acc':<10}{'Test Acc':<10}{'Time(min)':<11}{'Params'}",
        "-" * 68
    ]
    for r in results:
        lines.append(
            f"{r['label']:<8}"
            f"{'Yes' if r['use_pe'] else 'No':<6}"
            f"{r['num_heads']:<7}"
            f"{r['num_layers']:<8}"
            f"{r['val_acc']:<10}"
            f"{r['test_acc']:<10}"
            f"{r['train_time']:<11}"
            f"{r['params']:,}"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  saved table -> {path}")
