import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import argparse


def parse_args():
    p = argparse.ArgumentParser(description="Predict kcat using a trained model checkpoint.")
    p.add_argument("--ckpt", dest="ckpt_path", type=str, required=True,
                   help="Path to model checkpoint (.pth)")
    p.add_argument("--input", dest="data_csv", type=str, required=True,
                   help="CSV file to predict on (must contain required feature columns)")
    p.add_argument("--out", dest="out_csv", type=str, required=True,
                   help="Output file name for predictions")
    return p.parse_args()


args = parse_args()
CKPT_PATH = args.ckpt_path
UNSEEN_CSV = args.data_csv
OUT_CSV = args.out_csv


# -------- Columns (must match training) --------
feature_columns = [f'mol2vec_{i}' for i in range(300)] + \
                  [f'ec2vec_{i}' for i in range(1024)] + \
                  [f'Embedding_{i+1}' for i in range(128)]
target_column = 'kcat'  # optional; if present, metrics will be computed


# ---------------------- Model ----------------------
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=1,
                               out_channels=16,
                               kernel_size=5,
                               stride=1,
                               padding=0)
        self.bnc = nn.BatchNorm1d(16)

        # Transformer Encoder
        self.embedding_dim = 16  # Must match the CNN output channels
        encoder_layer = TransformerEncoderLayer(d_model=self.embedding_dim, nhead=4, dim_feedforward=32)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=2)

        output_length = self.output_dim(input_size)

        self.fc_input_dim = output_length * self.embedding_dim
        self.layer1 = nn.Linear(self.fc_input_dim, 2048)
        self.bn1 = nn.BatchNorm1d(2048)

        self.layer2 = nn.Linear(2048, 512)
        self.bn2 = nn.BatchNorm1d(512)

        self.layer3 = nn.Linear(512, 128)
        self.bn3 = nn.BatchNorm1d(128)

        self.output = nn.Linear(128, 1)

    def forward(self, x):
        x = x.unsqueeze(1)                 # (batch, 1, input_size)
        x = torch.relu(self.bnc(self.conv1(x)))  # (batch, 16, seq_len)

        # (seq_len, batch, emb_dim)
        x = x.permute(2, 0, 1)

        x = self.transformer_encoder(x)

        # flatten -> (batch, seq_len * emb_dim)
        x = x.permute(1, 0, 2).contiguous().view(x.shape[1], -1)

        x = torch.relu(self.bn1(self.layer1(x)))
        x = torch.relu(self.bn2(self.layer2(x)))
        x = torch.relu(self.bn3(self.layer3(x)))
        x = self.output(x)
        return x

    def output_dim(self, input_length):
        kernel_size = self.conv1.kernel_size[0]
        stride = self.conv1.stride[0]
        padding = self.conv1.padding[0]
        return (input_length - kernel_size + 2 * padding) // stride + 1


# -------- Load unseen data --------
df_unseen = pd.read_csv(UNSEEN_CSV)

missing = [c for c in feature_columns if c not in df_unseen.columns]
if missing:
    raise ValueError(
        f"Missing required feature columns in {UNSEEN_CSV}: {missing[:5]}{'...' if len(missing) > 5 else ''}"
    )

X_unseen = df_unseen[feature_columns].values.astype(np.float32)

# (Modification #2) Make labels optional
HAS_LABEL = (target_column in df_unseen.columns)
if HAS_LABEL:
    y_true = df_unseen[target_column].values.astype(np.float64)
    if np.any(y_true <= 0):
        raise ValueError("Found non-positive kcat values in labels; log10 transform requires y > 0.")
    y_true_log = np.log10(y_true).astype(np.float32)

    unseen_dataset = TensorDataset(
        torch.tensor(X_unseen, dtype=torch.float32),
        torch.tensor(y_true_log, dtype=torch.float32)
    )
else:
    unseen_dataset = TensorDataset(torch.tensor(X_unseen, dtype=torch.float32))

unseen_loader = DataLoader(unseen_dataset, batch_size=256, shuffle=False)

# -------- Rebuild model & load weights --------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size = X_unseen.shape[1]
model = NeuralNetwork(input_size).to(device)

# (Modification #3) Support both plain state_dict and dict checkpoints
ckpt = torch.load(CKPT_PATH, map_location=device)
state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
model.load_state_dict(state)
model.eval()

criterion = nn.MSELoss()

# -------- Inference --------
pred_log_list = []
true_log_list = []
loss_sum = 0.0
n_batches = 0

with torch.no_grad():
    for batch in unseen_loader:
        xb = batch[0].to(device)
        out = model(xb).view(-1)  # safer than squeeze; predicts log10(kcat)
        pred_log_list.append(out.detach().cpu().numpy())

        if HAS_LABEL:
            yb = batch[1].to(device).view(-1)
            loss = criterion(out, yb)
            loss_sum += loss.item()
            n_batches += 1
            true_log_list.append(yb.detach().cpu().numpy())

pred_log = np.concatenate(pred_log_list)

# -------- Metrics (only if labels exist) --------
if HAS_LABEL:
    true_log = np.concatenate(true_log_list)

    avg_mse_log = mean_squared_error(true_log, pred_log)
    r2_log = r2_score(true_log, pred_log)
    pear_log, _ = pearsonr(true_log, pred_log)
    avg_loss_log = loss_sum / max(n_batches, 1)

    pred_kcat = np.power(10.0, pred_log)
    true_kcat = np.power(10.0, true_log)

    mse_lin = mean_squared_error(true_kcat, pred_kcat)
    r2_lin = r2_score(true_kcat, pred_kcat)
    pear_lin, _ = pearsonr(true_kcat, pred_kcat)

    print(f"[LOG10]  AvgLoss={avg_loss_log:.6f}  MSE={avg_mse_log:.6f}  R2={r2_log:.4f}  Pearson={pear_log:.4f}")
    # print(f"[LINEAR] MSE={mse_lin:.6f}  R2={r2_lin:.4f}  Pearson={pear_lin:.4f}")
else:
    print("Ground-truth 'kcat' column not found in input CSV; skipping metrics and saving predictions only.")
    pred_kcat = np.power(10.0, pred_log)

# -------- Save predictions --------
# (Modification #1) Create output directory if needed
RESULTS_DIR = "Results"
os.makedirs(RESULTS_DIR, exist_ok=True)

OUT_FILENAME = os.path.basename(args.out_csv)   # ensure only filename
OUT_CSV = os.path.join(RESULTS_DIR, OUT_FILENAME)

out_df = df_unseen.copy()
out_df['pred_log10_kcat'] = pred_log
out_df['pred_kcat'] = pred_kcat
out_df.to_csv(OUT_CSV, index=False)
print(f"Saved predictions → {OUT_CSV}")
