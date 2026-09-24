import sys
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from sklearn.model_selection import train_test_split

sys.path.append(".")
from src.dataset import EssayDataset
from src.models.scorer import EssayScorer
from src.vocab import build_oov_vocab
from src.metrics import ThresholdOptimizer


def train_epoch(model, loader, opt, crit, device):
    model.train()
    tot_loss = 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        lbls = batch["labels"].to(device)

        opt.zero_grad()
        loss = crit(model(ids, mask), lbls)
        loss.backward()
        opt.step()
        tot_loss += loss.item()
    return tot_loss / len(loader)


def eval_epoch(model, loader, crit, device):
    model.eval()
    tot_loss = 0
    all_preds, all_lbls = [], []

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["labels"].to(device)

            preds = model(ids, mask)
            loss = crit(preds, lbls)
            tot_loss += loss.item()

            all_preds.extend(preds.cpu().numpy())
            all_lbls.extend(lbls.cpu().numpy())

    return tot_loss / len(loader), np.array(all_lbls), np.array(all_preds)


if __name__ == "__main__":
    csv_path = "data/train.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_list = build_oov_vocab(csv_path, top_n=500)
    full_ds = EssayDataset(csv_path, new_tokens=vocab_list)

    train_idx, val_idx = train_test_split(
        range(len(full_ds)), test_size=0.2, random_state=42
    )
    train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=8, shuffle=True)
    val_loader = DataLoader(Subset(full_ds, val_idx), batch_size=8, shuffle=False)

    model = EssayScorer(model_path="artifacts/mlm_model").to(device)
    opt = AdamW(model.parameters(), lr=2e-5)
    crit = nn.MSELoss()

    best_qwk = -1.0
    epochs = 5
    min_score, max_score = 1, 10

    for ep in range(epochs):
        t_loss = train_epoch(model, train_loader, opt, crit, device)
        v_loss, true_lbls, val_preds = eval_epoch(model, val_loader, crit, device)

        opt_rounder = ThresholdOptimizer(min_score, max_score)
        opt_coef = opt_rounder.fit(val_preds, true_lbls)
        val_qwk = -opt_rounder._loss(opt_coef, val_preds, true_lbls)

        print(
            f"Epoch {ep+1}/{epochs} | Tr MSE: {t_loss:.4f} | Val MSE: {v_loss:.4f} | QWK: {val_qwk:.4f}"
        )

        if val_qwk > best_qwk:
            best_qwk = val_qwk
            torch.save(model.state_dict(), "artifacts/final_scorer.pt")
            np.save("artifacts/thresholds.npy", opt_coef)
            print("  -> QWK improved. Weights & Thresholds saved.")

    print(f"Best QWK = {best_qwk:.4f}")
