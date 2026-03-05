# train.py
import argparse
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset

from model_for_github import NeuralNetwork


def natural_sort_key(s: str):
    """
    Sort keys like mol2vec_0, mol2vec_1, ..., mol2vec_10 numerically.
    Falls back to string sort if no trailing integer is found.
    """
    try:
        return (0, int(s.split("_")[-1]))
    except Exception:
        return (1, s)


def detect_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Auto-detect feature columns in the canonical order:
      mol2vec_* then ec2vec_* then Embedding_*

    This matches your original concatenation order (metabolite-EC-species).
    """
    mol_cols = [c for c in df.columns if c.startswith("mol2vec_")]
    ec_cols = [c for c in df.columns if c.startswith("ec2vec_")]
    emb_cols = [c for c in df.columns if c.startswith("Embedding_")]

    mol_cols = sorted(mol_cols, key=natural_sort_key)
    ec_cols = sorted(ec_cols, key=natural_sort_key)
    emb_cols = sorted(emb_cols, key=natural_sort_key)

    feature_columns = mol_cols + ec_cols + emb_cols

    if len(feature_columns) == 0:
        raise ValueError(
            "No feature columns found. Expected columns starting with "
            "'mol2vec_', 'ec2vec_', and/or 'Embedding_'."
        )

    print("Detected feature columns:")
    print(f"  mol2vec_*   : {len(mol_cols)}")
    print(f"  ec2vec_*    : {len(ec_cols)}")
    print(f"  Embedding_* : {len(emb_cols)}")
    print(f"  TOTAL       : {len(feature_columns)}")

    # Helpful warning if one block is missing (still allowed)
    if len(mol_cols) == 0:
        print("WARNING: No mol2vec_* columns detected.")
    if len(ec_cols) == 0:
        print("WARNING: No ec2vec_* columns detected.")
    if len(emb_cols) == 0:
        print("WARNING: No Embedding_* columns detected.")

    return feature_columns


def make_loaders(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    return train_loader, val_loader


def parse_args():
    p = argparse.ArgumentParser(
        description="Train CNN+Transformer kcat predictor (single train/val split, no pseudo data)."
    )

    p.add_argument("--data_csv", type=str, required=True, help="CSV path.")
    p.add_argument("--target_column", type=str, default="kcat")

    p.add_argument("--save_dir", type=str, default="saved_NN_models")
    p.add_argument("--ckpt_name", type=str, default="best_model.pth")

    p.add_argument("--val_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=5e-3)

    p.add_argument("--no_cuda", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    print(f"Using device: {device}")

    # reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df = pd.read_csv(args.data_csv)

    # auto-detect features in metabolite-EC-species order
    feature_columns = detect_feature_columns(df)

    # required target
    if args.target_column not in df.columns:
        raise KeyError(f"Target column '{args.target_column}' not found in CSV.")

    # drop NaNs in features + target
    df = df.dropna(subset=feature_columns + [args.target_column]).reset_index(drop=True)

    # split
    train_df, val_df = train_test_split(
        df, test_size=args.val_ratio, random_state=args.seed, shuffle=True
    )
    print(f"Split: train={len(train_df)}  val={len(val_df)}")

    # numpy arrays
    X_train = train_df[feature_columns].values
    y_train = train_df[args.target_column].astype(np.float64).values

    X_val = val_df[feature_columns].values
    y_val = val_df[args.target_column].astype(np.float64).values

    # log10 transform (consistent with your original code)
    # NOTE: kcat must be > 0
    if np.any(y_train <= 0) or np.any(y_val <= 0):
        raise ValueError("Found non-positive kcat values. log10 requires kcat > 0.")

    y_train_log = np.log10(y_train)
    y_val_log = np.log10(y_val)

    train_loader, val_loader = make_loaders(X_train, y_train_log, X_val, y_val_log, args.batch_size)

    # model
    input_size = X_train.shape[1]
    model = NeuralNetwork(input_size).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, args.ckpt_name)

    best_val_loss = float("inf")
    best_metrics = None

    for epoch in range(args.epochs):
        # train
        model.train()
        train_loss_sum = 0.0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(xb).squeeze()
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.detach().cpu().item()

        avg_train_loss = train_loss_sum / max(len(train_loader), 1)

        # val
        model.eval()
        val_loss_sum = 0.0
        preds = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                pred = model(xb).squeeze()
                loss = criterion(pred, yb)
                val_loss_sum += loss.detach().cpu().item()
                preds.extend(pred.detach().cpu().numpy())

        avg_val_loss = val_loss_sum / max(len(val_loader), 1)

        y_pred = np.array(preds, dtype=np.float64)
        mse = mean_squared_error(y_val_log, y_pred)
        r2 = r2_score(y_val_log, y_pred)
        pear = pearsonr(y_val_log, y_pred)[0]

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"epoch {epoch+1:4d}/{args.epochs}  "
                f"train_loss={avg_train_loss:.6f}  val_loss={avg_val_loss:.6f}  "
                f"MSE={mse:.6f}  R2={r2:.6f}  Pearson={pear:.6f}"
            )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_metrics = (mse, r2, pear)
            torch.save(model.state_dict(), ckpt_path)

    print("\n=== Best checkpoint (by lowest val loss) ===")
    print(f"Saved to: {ckpt_path}")
    if best_metrics is not None:
        mse, r2, pear = best_metrics
        print(f"Val MSE: {mse:.6f}")
        print(f"Val R2: {r2:.6f}")
        print(f"Val Pearson: {pear:.6f}")


if __name__ == "__main__":
    main()