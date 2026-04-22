# Mini Transformer Benchmark — CPE663

A mini Transformer encoder built completely from scratch in PyTorch.
No prebuilt Transformer modules were used anywhere in this project.

---

## What is this project?

This project builds a small Transformer model and tests it on a simple sequence task.
The goal is to understand how each part of the Transformer contributes to performance
by comparing four different model configurations.

---

## What does the model do?

The model looks at a sequence of tokens and answers one yes/no question:

> Does the **first token** appear again in the **second half** of the sequence?

**Example:**
```
Sequence : [A, C, B, D, A, PAD, PAD]
First token        : A
Second half tokens : [D, A]
A appears in second half → label = 1 (yes)
```

**Another example:**
```
Sequence : [B, C, D, A, C, PAD, PAD]
First token        : B
Second half tokens : [A, C]
B does not appear  → label = 0 (no)
```

---

## Vocabulary

| Symbol | ID |
|--------|----|
| PAD    | 0  |
| A      | 1  |
| B      | 2  |
| C      | 3  |
| D      | 4  |

PAD is just a filler token used to make all sequences the same length (20).

---

## Project Structure

```
mini_transformer_benchmark/
│
├── data.py           → loads the CSV files into PyTorch DataLoaders
├── model.py          → the Transformer built from scratch
├── train.py          → trains one model and returns results
├── benchmark.py      → runs all 4 models and compares them
├── utils.py          → saves plots and prints the results table
│
├── report.pdf        → written report
├── README.md         → this file
│
└── data/
    ├── train.csv       (5,000 samples)
    ├── validation.csv  (1,000 samples)
    └── test.csv        (1,000 samples)
```

---

## How the model works (simple version)

```
Input tokens  →  [A, C, B, D, A, PAD, PAD]
      ↓
Token Embedding   →  each token becomes a vector of 64 numbers
      ↓
Positional Encoding  →  adds position info so the model knows token order
      ↓
Encoder Block(s)   →  attention lets every token look at every other token
                       feed-forward network transforms each token
      ↓
Mean Pooling   →  average all real token vectors into one vector
      ↓
Classifier     →  two scores → pick the higher one → 0 or 1
```

---

## Requirements

Python 3.8 or higher is required.

Install dependencies:
```bash
pip install torch matplotlib
```

---

## How to run

**Step 1 — Put your data files in the right place:**
```
mini_transformer_benchmark/
└── data/
    ├── train.csv
    ├── validation.csv
    └── test.csv
```

**Step 2 — Run the full benchmark (trains all 4 models):**
```bash
python benchmark.py
```

**Or train just one model to test quickly:**
```bash
python train.py
```

---

## What gets created after running

| File | What it is |
|------|------------|
| `model_A.pt` | Saved weights for Model A |
| `model_B.pt` | Saved weights for Model B |
| `model_C.pt` | Saved weights for Model C |
| `model_D.pt` | Saved weights for Model D |
| `curve_A.png` | Training curve plot for Model A |
| `curve_B.png` | Training curve plot for Model B |
| `curve_C.png` | Training curve plot for Model C |
| `curve_D.png` | Training curve plot for Model D |
| `benchmark_results.txt` | Final results table |

---

## The 4 model variants

Each model changes one thing to see how it affects performance:

| Model | What is different | Purpose |
|-------|-------------------|---------|
| A | 1 attention head | baseline with positional encoding |
| B | 4 attention heads | does more heads help? |
| C | no positional encoding | does position info matter? |
| D | 2 encoder layers | does going deeper help? |

---

## Results

| Model | PE  | Heads | Layers | Val Acc | Test Acc | Time  | Params |
|-------|-----|-------|--------|---------|----------|-------|--------|
| A     | Yes | 1     | 1      | 0.929   | 0.914    | 0.26m | 33,922 |
| B     | Yes | 4     | 1      | 0.954   | 0.941    | 0.31m | 33,922 |
| C     | No  | 4     | 1      | 0.814   | 0.823    | 0.25m | 33,922 |
| D     | Yes | 4     | 2      | 0.987   | 0.980    | 0.46m | 67,394 |

**The majority class baseline is ~0.823** — a model that always guesses 1 gets this score.
Any model below or at this score has learned nothing.

---

## Key findings

**Model C (no positional encoding) = completely failed**
The model never improved past 0.823 across all 15 epochs.
Without position info, the model cannot tell which token is first
or where the second half starts — the task is impossible for it.

**Model B (4 heads) > Model A (1 head)**
Same number of parameters, but 4 heads gives better accuracy and smoother training.
Each head can specialise in a different aspect of the task.

**Model D (2 layers) = best overall**
The second encoder layer builds on the first, allowing richer understanding.
Achieved 0.980 test accuracy — 15.7 points above the baseline.

---

## Implementation notes

All of the following were built from scratch — no torch.nn.Transformer,
no torch.nn.MultiheadAttention, no HuggingFace models:

- Token embedding
- Sinusoidal positional encoding
- Scaled dot-product attention with padding mask
- Multi-head self-attention
- Position-wise feed-forward network
- Residual connections
- Layer normalisation
- Mean pooling over non-PAD tokens

Random seed is fixed at 42 for reproducibility.
