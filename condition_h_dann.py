"""Condition H: domain-adversarial neural network for tabular adaptation."""

from __future__ import annotations

import argparse
import copy
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.autograd import Function

from advanced_features import build_domain_invariant_features, fit_region_age_stats
from pu_learning import ElkanNotoPU, require_observation_mask
from top_k_metrics import top_k_metrics, top_k_recall

RANDOM_STATE = 42
TARGET = "chronic_homeless"


class GradientReversal(Function):
    @staticmethod
    def forward(ctx, values, coefficient):
        ctx.coefficient = coefficient
        return values.view_as(values)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.coefficient * gradient, None


def gradient_reversal(values, coefficient):
    return GradientReversal.apply(values, coefficient)


class DANN(nn.Module):
    def __init__(self, n_features: int, n_domains: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 32), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(32, 16), nn.ReLU(),
        )
        self.task_head = nn.Linear(16, 1)
        self.domain_head = nn.Linear(16, n_domains)

    def forward(self, features, coefficient):
        hidden = self.encoder(features)
        task_logits = self.task_head(hidden).squeeze(1)
        domain_logits = self.domain_head(gradient_reversal(hidden, coefficient))
        return task_logits, domain_logits


def load_and_tag(path, source_name, data_type):
    frame = pd.read_csv(path)
    frame["data_source"] = source_name
    frame["data_type"] = data_type
    return frame


def main(toronto_path, ottawa_path, lanark_path, val_size=0.2,
         epochs=50, learning_rate=1e-3, domain_weight=0.2,
         monitor="auc_pr", patience=8, min_delta=1e-4,
         pu_c_mental_health=None, pu_c_substance_use=None,
         output_dir="predictions"):
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    toronto = load_and_tag(toronto_path, "toronto_synth", "synthetic")
    ottawa = load_and_tag(ottawa_path, "ottawa_synth", "synthetic")
    lanark = load_and_tag(lanark_path, "real_lanark", "observed")
    lanark_train, lanark_val = train_test_split(
        lanark, test_size=val_size, stratify=lanark[TARGET], random_state=RANDOM_STATE
    )

    train_pool = pd.concat([lanark_train, toronto, ottawa], ignore_index=True)
    age_stats = fit_region_age_stats(train_pool)
    train_features = build_domain_invariant_features(train_pool, age_stats)
    val_features = build_domain_invariant_features(lanark_val, age_stats)

    # PU-derived columns are opt-in because current cleaned files lack the
    # observation masks needed to distinguish unlabeled from negative values.
    for variable, c in (("mental_health", pu_c_mental_health),
                        ("substance_use", pu_c_substance_use)):
        if c is None:
            continue
        observed = require_observation_mask(train_pool, variable)
        pu_model = ElkanNotoPU(LogisticRegression(max_iter=1000), c=float(c))
        pu_model.fit(train_features, observed)
        train_features[f"pu_{variable}_risk"] = pu_model.predict_proba(train_features)[:, 1]
        val_features[f"pu_{variable}_risk"] = pu_model.predict_proba(val_features)[:, 1]

    x_train = train_features.to_numpy(dtype=np.float32)
    x_val = val_features.to_numpy(dtype=np.float32)
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train).astype(np.float32)
    x_val = scaler.transform(x_val).astype(np.float32)

    y_train = train_pool[TARGET].astype(float).to_numpy(dtype=np.float32)
    y_val = lanark_val[TARGET].astype(int).to_numpy()
    domain_names = {name: index for index, name in enumerate(sorted(train_pool["data_source"].unique()))}
    domains = train_pool["data_source"].map(domain_names).to_numpy(dtype=np.int64)

    model = DANN(x_train.shape[1], len(domain_names))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    task_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(
        [(y_train == 0).sum() / max((y_train == 1).sum(), 1)], dtype=torch.float32
    ))
    domain_loss = nn.CrossEntropyLoss()
    x_tensor = torch.from_numpy(x_train)
    y_tensor = torch.from_numpy(y_train)
    domain_tensor = torch.from_numpy(domains.copy())

    best_score = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(epochs):
        progress = epoch / max(epochs - 1, 1)
        coefficient = domain_weight * (2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0)
        optimizer.zero_grad()
        task_logits, domain_logits = model(x_tensor, float(coefficient))
        loss = task_loss(task_logits, y_tensor) + domain_loss(domain_logits, domain_tensor)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_logits, _ = model(torch.from_numpy(x_val), 0.0)
            validation_proba = torch.sigmoid(validation_logits).numpy()
        if monitor == "auc_pr":
            score = float(average_precision_score(y_val, validation_proba))
        else:
            score = float(top_k_metrics(y_val, validation_proba, 0.10)["precision_at_k"])
        if score > best_score + min_delta:
            best_score = score
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            epochs_without_improvement += 1
        model.train()
        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch + 1}; best {monitor}={best_score:.4f} "
                  f"at epoch {best_epoch}.")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_logits, _ = model(torch.from_numpy(x_val), 0.0)
        proba = torch.sigmoid(val_logits).numpy()

    metrics = {
        "auc_pr": float(average_precision_score(y_val, proba)),
        "monitor": monitor,
        "best_monitor": best_score,
        "best_epoch": best_epoch,
        "top_5": top_k_metrics(y_val, proba, 0.05),
        "top_10": top_k_metrics(y_val, proba, 0.10),
        "top_20": top_k_metrics(y_val, proba, 0.20),
        **top_k_recall(y_val, proba),
    }
    print(json.dumps(metrics, indent=2))

    os.makedirs(output_dir, exist_ok=True)
    pd.DataFrame({
        "lanark_idx": lanark_val.index,
        "y_true": y_val,
        "proba": proba,
    }).to_csv(os.path.join(output_dir, "H_dann.csv"), index=False)
    with open(os.path.join(output_dir, "H_dann_top_k.json"), "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Condition H: domain-adversarial tabular neural network")
    parser.add_argument("--toronto", required=True)
    parser.add_argument("--ottawa", required=True)
    parser.add_argument("--lanark", required=True)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=50,
                        help="Maximum epochs; 20-50 is recommended with early stopping")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--domain_weight", type=float, default=0.2)
    parser.add_argument("--monitor", choices=["auc_pr", "precision_at_10"], default="auc_pr")
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--pu_c_mental_health", type=float, default=None)
    parser.add_argument("--pu_c_substance_use", type=float, default=None)
    parser.add_argument("--output_dir", default="predictions")
    args = parser.parse_args()
    main(
        toronto_path=args.toronto,
        ottawa_path=args.ottawa,
        lanark_path=args.lanark,
        val_size=args.val_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        domain_weight=args.domain_weight,
        monitor=args.monitor,
        patience=args.patience,
        min_delta=args.min_delta,
        pu_c_mental_health=args.pu_c_mental_health,
        pu_c_substance_use=args.pu_c_substance_use,
        output_dir=args.output_dir,
    )
