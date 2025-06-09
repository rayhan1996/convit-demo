

# main.py – ConViT training with class‑wise thresholds & partial fine‑tuning
# -----------------------------------------------------------------------------
# • loads 4‑channel dataset (TIFF)  ➜  multi‑label (clouds, shadows, land‑cover)
# • ConViT = EfficientNet‑B0 (4‑ch)  +  ViT‑B/16 fusion
# • unfreezes *only* the last two EfficientNet blocks for fine‑tuning
# • tracks best F1_micro using *per‑class* thresholds and saves checkpoint
# -----------------------------------------------------------------------------

import torch, numpy as np, pandas as pd, shutil, random, os
import torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score
from pathlib import Path

from data.dataset      import SatImageMultiLabelDataset
from models.hybrid_vit import ConViT

# reproducibility --------------------------------------------------------------
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# helper functions -------------------------------------------------------------
_SIGMOID = lambda x: 1. / (1. + np.exp(-x))

def per_class_thresholds(logits: np.ndarray, labels: np.ndarray, step: float = .01):
    """Grid‑search best threshold *per class* on validation set."""
    if logits.min() < 0 or logits.max() > 1:
        logits = _SIGMOID(logits)
    best_t = np.zeros(logits.shape[1])
    for c in range(logits.shape[1]):
        ts = np.arange(0, 1 + step, step)
        best_t[c] = max(ts, key=lambda t: f1_score(labels[:, c], (logits[:, c] >= t)))
    preds = (logits >= best_t).astype(int)
    return best_t, f1_score(labels, preds, average="macro")

@torch.no_grad()
def evaluate(model, loader, thr, device):
    """Return F1_micro, F1_macro, Accuracy using threshold vector *thr*."""
    model.eval(); P, Y = [], []
    for X, y in loader:
        P.append(torch.sigmoid(model(X.to(device))).cpu().numpy())
        Y.append(y.numpy())
    P, Y = np.concatenate(P), np.concatenate(Y)
    preds = (P >= thr).astype(int)
    return (
        f1_score(Y, preds, average="micro"),
        f1_score(Y, preds, average="macro"),
        accuracy_score(Y, preds),
    )

@torch.no_grad()
def gather_logits(model, loader, device):
    L, Y = [], []
    for X, y in loader:
        L.append(model(X.to(device)).cpu())
        Y.append(y.cpu())
    return torch.cat(L), torch.cat(Y)

# paths -----------------------------------------------------------------------
CKPT_DIR  = Path("/content/drive/MyDrive/satellite images detection/sat_hybrid_checkpoints")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_FILE = CKPT_DIR / "hybrid_ckpt.pt"

# data ------------------------------------------------------------------------
DATA_DIR  = Path("/content/dataset")
IMG_DIR   = DATA_DIR / "Images" / "Images"
train_df  = pd.read_csv(DATA_DIR / "train.csv")
val_df    = pd.read_csv(DATA_DIR / "validation.csv")

train_loader = DataLoader(
    SatImageMultiLabelDataset(train_df, IMG_DIR),
    batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(
    SatImageMultiLabelDataset(val_df, IMG_DIR),
    batch_size=16, shuffle=False, num_workers=2)

NUM_CLASSES = train_df.shape[1] - 1
class_names = list(train_df.columns[1:])

# model -----------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model  = ConViT(patch_size=16, embed_dim=768, num_classes=NUM_CLASSES).to(device)

# unfreeze last 2 EfficientNet blocks (6 & 7) ---------------------------------
for name, p in model.cnn.named_parameters():
    if "blocks.6" in name or "blocks.7" in name:
        p.requires_grad = True
    else:
        p.requires_grad = False
# ViT & classifier params already require_grad=True

# loss & optimizer ------------------------------------------------------------
label_sum  = train_df.iloc[:, 1:].sum().values + 1e-6
pos_weight = torch.tensor((len(train_df) - label_sum) / label_sum,
                          dtype=torch.float32, device=device)
# bare_groud
bare_idx = class_names.index("bare_ground"); pos_weight[bare_idx] *= 1.5

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

backbone_params = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("cnn.")]
head_params     = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("cnn.")]

optimizer = optim.AdamW([
        {"params": head_params,     "lr": 1e-3},
        {"params": backbone_params, "lr": 1e-4},
    ], weight_decay=0.05)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

# resume ----------------------------------------------------------------------
start_epoch, best_micro_f1, best_thr = 0, 0.0, np.full(NUM_CLASSES, 0.5)
if CKPT_FILE.exists():
    ckpt = torch.load(CKPT_FILE, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    best_micro_f1 = float(ckpt["best_f1"])
    best_thr      = np.array(ckpt["best_thr"], dtype=np.float32)
    start_epoch   = int(ckpt["epoch"]) + 1
    print(f"🔄  Resumed at epoch {start_epoch} | best F1_micro = {best_micro_f1:.4f}")
else:
    print("🆕  Training from scratch")

# training loop ---------------------------------------------------------------
EPOCHS = start_epoch + 20
for epoch in range(start_epoch, EPOCHS):
    model.train(); total_loss = 0.0; seen = 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device).float()
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        seen       += X.size(0)

    scheduler.step()
    train_loss = total_loss / seen

    # evaluation with *current* best_thr -------------------------------------
    f1_micro, f1_macro, acc = evaluate(model, val_loader, best_thr, device)

    # recompute optimal thresholds on logits ---------------------------------
    logits, labels = gather_logits(model, val_loader, device)
    thr_cls, _ = per_class_thresholds(logits.numpy(), labels.numpy())

    # checkpoint if improved --------------------------------------------------
    if f1_micro > best_micro_f1:
        best_micro_f1, best_thr = f1_micro, thr_cls
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "best_f1": best_micro_f1,
            "best_thr": best_thr,
        }, CKPT_FILE)
        print(f"✅  New best F1_micro = {best_micro_f1:.4f}  (saved)")

    # logging ----------------------------------------------------------------
    lr_now = optimizer.param_groups[0]['lr']
    print(f"Epoch [{epoch+1}/{EPOCHS}]  "
          f"Loss: {train_loss:.4f} | "
          f"F1_micro: {f1_micro:.4f} | "
          f"F1_macro: {f1_macro:.4f} | "
          f"Acc: {acc:.4f} | "
          f"LR: {lr_now:.6f}")

print(f"🏁  Training finished | Best F1_micro = {best_micro_f1:.4f}")
