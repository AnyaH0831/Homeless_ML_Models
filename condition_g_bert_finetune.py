"""
Condition G: BERT-base (full model, not DistilBERT) pretrain-then-finetune

Stage 1 (pretrain): Fine-tune FULL bert-base-uncased (all encoder layers +
                     classification head) on the serialized Toronto+Ottawa
                     text pool.

Stage 2 (finetune):  UNFREEZE the last two transformer encoder layers (Layers 10 and 11)
                     + the classification head. Train them on real Lanark's
                     serialized text. Unfreezing deeper layers allows the model
                     to adapt better to the target domain.

Saves the final trained model and tokenizer to disk upon completion.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, classification_report
)
from transformers import BertTokenizerFast, BertForSequenceClassification, get_linear_schedule_with_warmup

RANDOM_STATE = 42
TARGET = "chronic_homeless"
MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 96

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

class SerializedTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=MAX_LENGTH):
        self.texts = texts.tolist() if hasattr(texts, "tolist") else list(texts)
        self.labels = torch.tensor(labels.values if hasattr(labels, "values") else labels, dtype=torch.long)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        item = {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten()
        }
        item["labels"] = self.labels[idx]
        return item

def train_epoch(model, loader, optimizer, scheduler, device, class_weights=None):
    model.train()
    total_loss = 0.0
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    for batch in loader:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = loss_fn(outputs.logits, labels)
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)

def evaluate_model(model, loader, device):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(outputs.logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    auc_roc = roc_auc_score(all_labels, all_probs)
    auc_pr = average_precision_score(all_labels, all_probs)
    brier = brier_score_loss(all_labels, all_probs)

    return auc_roc, auc_pr, brier, all_preds, all_probs, all_labels

def save_predictions(lanark_indices, labels, probs, preds, condition_name, model_name, out_dir="predictions"):
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame({
        "lanark_idx": lanark_indices,
        "y_true": labels,
        "proba": probs,
        "predicted_class": preds,
    }).to_csv(os.path.join(out_dir, f"{condition_name}_{model_name}.csv"), index=False)

def main(toronto_path, ottawa_path, lanark_path, val_size=0.2,
         pretrain_epochs=8, pretrain_lr=2e-5, pretrain_batch_size=16,
         finetune_epochs=20, finetune_lr=2e-5, finetune_batch_size=16,
         save_model_dir="saved_models/condition_g_bert"):

    device = get_device()
    print(f"Using device: {device}")

    tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

    # =========================================================
    # STAGE 1: PRE-TRAIN ON SYNTHETIC POOL (TORONTO + OTTAWA)
    # =========================================================
    print("Loading synthetic datasets for Stage 1 pretraining...")
    toronto = pd.read_csv(toronto_path)
    ottawa = pd.read_csv(ottawa_path)

    pretrain_pool = pd.concat([toronto, ottawa], ignore_index=True)
    print(f"Pretrain pool combined size: {pretrain_pool.shape[0]} rows.")

    pretrain_dataset = SerializedTextDataset(
        pretrain_pool["text"], pretrain_pool[TARGET], tokenizer, max_len=MAX_LENGTH
    )
    pretrain_loader = DataLoader(pretrain_dataset, batch_size=pretrain_batch_size, shuffle=True)

    model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)

    # Enable gradients across the full model for pretraining
    for param in model.parameters():
        param.requires_grad = True

    n_pos = pretrain_pool[TARGET].sum()
    n_neg = len(pretrain_pool) - n_pos
    class_weights = torch.tensor([1.0, n_neg / max(n_pos, 1)], dtype=torch.float32).to(device)

    optimizer = AdamW(model.parameters(), lr=pretrain_lr, weight_decay=0.01)
    total_steps = len(pretrain_loader) * pretrain_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )

    for epoch in range(pretrain_epochs):
        loss = train_epoch(model, pretrain_loader, optimizer, scheduler, device, class_weights)
        print(f"  Pretrain epoch {epoch + 1}/{pretrain_epochs}: loss = {loss:.4f}")
    print("Stage 1 (pretrain) complete: full BERT-base fine-tuned on synthetic pool.")

    # Reference - zero-shot prediction
    lanark_full = pd.read_csv(lanark_path)
    lanark_train, lanark_val = train_test_split(
        lanark_full, test_size=val_size, random_state=RANDOM_STATE, stratify=lanark_full[TARGET]
    )
    lanark_val_dataset = SerializedTextDataset(lanark_val["text"], lanark_val[TARGET], tokenizer, max_len=MAX_LENGTH)
    lanark_val_loader = DataLoader(lanark_val_dataset, batch_size=finetune_batch_size, shuffle=False)

    # =========================================================
    # STAGE 2: FINE-TUNE -- UNFREEZE LAYERS 10, 11 + CLASSIFIER HEAD
    # =========================================================
    for name, param in model.named_parameters():
        if "classifier" in name or "layer.11" in name or "layer.10" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    model.config.hidden_dropout_prob = 0.1
    model.config.attention_probs_dropout_prob = 0.1

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"Fine-tuning active parameters: {len(trainable)} parameters specified to train.")

    lanark_train_dataset = SerializedTextDataset(
        lanark_train["text"], lanark_train[TARGET], tokenizer, max_len=MAX_LENGTH
    )
    lanark_train_loader = DataLoader(lanark_train_dataset, batch_size=finetune_batch_size, shuffle=True)

    n_pos_lanark = lanark_train[TARGET].sum()
    n_neg_lanark = len(lanark_train) - n_pos_lanark
    lanark_class_weights = torch.tensor(
        [1.0, n_neg_lanark / max(n_pos_lanark, 1)], dtype=torch.float32
    ).to(device)

    finetune_params = [p for p in model.parameters() if p.requires_grad]
    finetune_optimizer = AdamW(finetune_params, lr=finetune_lr, weight_decay=0.01)

    ft_total_steps = len(lanark_train_loader) * finetune_epochs
    ft_scheduler = get_linear_schedule_with_warmup(
        finetune_optimizer, num_warmup_steps=int(0.1 * ft_total_steps), num_training_steps=ft_total_steps
    )

    for epoch in range(finetune_epochs):
        loss = train_epoch(model, lanark_train_loader, finetune_optimizer, ft_scheduler, device, lanark_class_weights)
        print(f"  Finetune epoch {epoch + 1}/{finetune_epochs}: loss = {loss:.4f}")
    print(f"Stage 2 (finetune) complete: {finetune_epochs} epochs.")

    # =========================================================
    # EVALUATION ON HELD-OUT REAL LANARK
    # =========================================================
    auc_roc, auc_pr, brier, preds, probs, labels = evaluate_model(model, lanark_val_loader, device)
    print(f"\nEvaluation results on Lanark test partition ({len(lanark_val)} samples):")
    print(f"  AUC-ROC: {auc_roc:.4f}")
    print(f"  AUC-PR : {auc_pr:.4f}")
    print(f"  Brier  : {brier:.4f}")

    print("\nClassification Report:")
    print(classification_report(labels, preds, target_names=["Not Chronic", "Chronic"]))

    # Save predictions
    save_predictions(lanark_val.index, labels, probs, preds, "G", "bert")
    print("Predictions saved to predictions/G_bert.csv.")

    # Save fine-tuned checkpoint
    os.makedirs(save_model_dir, exist_ok=True)
    model.save_pretrained(save_model_dir)
    tokenizer.save_pretrained(save_model_dir)
    print(f"Saved final model checkpoint and tokenizer to {save_model_dir}")

if __name__ == "__main__":
    is_notebook = any("kernel" in arg for arg in sys.argv)

    if len(sys.argv) > 1 and not is_notebook:
        parser = argparse.ArgumentParser(description="Condition G: BERT Pretrain & Fine-tune Pipeline")
        parser.add_argument("--toronto", required=True, help="Path to SERIALIZED Toronto CSV (from serialize.py)")
        parser.add_argument("--ottawa", required=True, help="Path to SERIALIZED Ottawa CSV")
        parser.add_argument("--lanark", required=True, help="Path to SERIALIZED Lanark CSV")
        parser.add_argument("--val_size", type=float, default=0.2)
        parser.add_argument("--pretrain_epochs", type=int, default=8)
        parser.add_argument("--pretrain_lr", type=float, default=2e-5)
        parser.add_argument("--pretrain_batch_size", type=int, default=16)
        parser.add_argument("--finetune_epochs", type=int, default=20)
        parser.add_argument("--finetune_lr", type=float, default=2e-5)
        parser.add_argument("--finetune_batch_size", type=int, default=16)
        parser.add_argument("--save_model_dir", type=str, default="saved_models/condition_g_bert")
        args = parser.parse_args()

        main(
            toronto_path=args.toronto,
            ottawa_path=args.ottawa,
            lanark_path=args.lanark,
            val_size=args.val_size,
            pretrain_epochs=args.pretrain_epochs,
            pretrain_lr=args.pretrain_lr,
            pretrain_batch_size=args.pretrain_batch_size,
            finetune_epochs=args.finetune_epochs,
            finetune_lr=args.finetune_lr,
            finetune_batch_size=args.finetune_batch_size,
            save_model_dir=args.save_model_dir
        )
    else:
        print("Notebook environment detected. Running inside notebook using absolute paths:")
        toronto_p = "/data/serialized_toronto_sasm.csv"
        ottawa_p = "/data/serialized_ottawa_sasm.csv"
        lanark_p = "/data/serialized_real_lanark.csv"
        main(toronto_path=toronto_p, ottawa_path=ottawa_p, lanark_path=lanark_p)
