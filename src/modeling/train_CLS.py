"""
A script for training and evaluating a multilabel text 
classification model using PyTorch and Transformers (using single CLS embedding)
with extended metrics (precision, recall, F1 - micro and macro).

This script performs the following tasks:
1. Reads pipeline parameters and data paths.
2. Prepares and tokenizes the training and validation datasets.
3. Initializes a transformer-based classification model with a custom head that uses [CLS] token embedding.
4. Trains the model over multiple epochs, logging performance metrics (loss, precision, recall, F1 scores).
5. Evaluates the model on the validation dataset.
6. Saves the trained model and tokenizer for future use.
7. Saves a metrics history table as a CSV file.

Classes:
    CustomDataset: A dataset class for tokenizing text data and handling labels.
    CLSClassifier: A classification model with a transformer backbone and a custom classifier head.

Functions:
    train_epoch: Trains the model for one epoch and computes average loss and metrics.
    evaluate: Evaluates the model on the validation dataset and computes average loss and metrics.

Usage:
    Run the script with `typer`:
    ```
    python train.py --params-path <path_to_params_file>
    ```
"""

import sys
from pathlib import Path
from transformers import AutoModel, AutoTokenizer, AdamW
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score
)
import numpy as np
from tqdm import tqdm
import pandas as pd
import typer
from loguru import logger

sys.path.append(str(Path(__file__).resolve().parent.parent))

from entities.params import read_pipeline_params

app = typer.Typer()

project_root = Path().resolve().parent

log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)

log_file = log_dir / "train_cls_extended_metrics.log"
logger.add(log_file, rotation="10 MB", retention="10 days", level="INFO")

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CustomDataset(Dataset):
    """
    A Dataset for tokenizing text data with labels.

    Args:
        texts (list[str]): Input texts.
        labels (list): Labels for the texts.
        tokenizer: Tokenizer for text processing.
        max_length (int): Max sequence length (default: 128).
    """

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(),
            "attention_mask": encoded["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.float32),
        }


class CLSClassifier(nn.Module):
    """
    A classification model with a transformer base and a custom classifier head.

    Args:
        base_model: Pretrained transformer model.
        num_classes (int): Number of output classes.
    """

    def __init__(self, base_model, num_classes):
        super(CLSClassifier, self).__init__()
        self.base_model = base_model
        self.classifier = nn.Sequential(
            nn.Linear(base_model.config.hidden_size, base_model.config.hidden_size),
            nn.Tanh(),
            nn.Linear(base_model.config.hidden_size, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        return self.classifier(cls_output)


def compute_metrics(labels, preds):
    """
    Computes precision, recall, and F1 scores (micro and macro averages).

    Args:
        labels: True labels.
        preds: Predicted labels.

    Returns:
        dict: Dictionary containing all computed metrics.
    """
    metrics = {
        "precision_micro": precision_score(labels, preds, average="micro", zero_division=0),
        "recall_micro": recall_score(labels, preds, average="micro", zero_division=0),
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
    }
    return metrics


def train_epoch(model, dataloader, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model: The model to train.
        dataloader: DataLoader for training data.
        optimizer: Optimizer for updating model weights.
        device: Device for computation (CPU/GPU).

    Returns:
        tuple: Average loss and metrics for the epoch.
    """
    model.train()
    losses = []
    all_preds = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Training"):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask)
        loss = nn.BCEWithLogitsLoss()(outputs, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        preds = (torch.sigmoid(outputs).cpu().detach().numpy() > 0.5).astype(int)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = np.mean(losses)
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    return avg_loss, metrics


def evaluate(model, dataloader, device):
    """
    Evaluates the model on a validation dataset.

    Args:
        model: The model to evaluate.
        dataloader: DataLoader for validation data.
        device: Device for computation (CPU/GPU).

    Returns:
        tuple: Average loss and metrics for the validation dataset.
    """
    model.eval()
    all_preds = []
    all_labels = []
    losses = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            loss = nn.BCEWithLogitsLoss()(outputs, labels)
            losses.append(loss.item())

            preds = (torch.sigmoid(outputs).cpu().detach().numpy() > 0.5).astype(int)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = np.mean(losses)
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    return avg_loss, metrics


@app.command()
def main(params_path: str) -> None:
    """
    Main function to train and evaluate the text classification model.

    Args:
        params_path (str): Path to the pipeline parameters file.
    """
    params = read_pipeline_params(params_path)
    set_seed(params.experiment.seed)
    train_data = pd.read_csv(Path(params.paths.processed_data_dir) / params.data.train_file)
    val_data = pd.read_csv(Path(params.paths.processed_data_dir) / params.data.val_file)

    train_data = train_data.dropna(subset=["comment_text"])
    val_data = val_data.dropna(subset=["comment_text"])

    train_texts = train_data["comment_text"].tolist()
    train_labels = train_data[
        ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    ].values

    val_texts = val_data["comment_text"].tolist()
    val_labels = val_data[
        ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    ].values

    tokenizer = AutoTokenizer.from_pretrained(params.model.pretrained_model_name)
    base_model = AutoModel.from_pretrained(params.model.pretrained_model_name)

    num_classes = train_labels.shape[1]
    model = CLSClassifier(base_model, num_classes)
    optimizer = AdamW(model.parameters(), lr=params.experiment.learning_rate)

    train_dataset = CustomDataset(train_texts, train_labels, tokenizer, max_length=params.experiment.max_seq_length)
    val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_length=params.experiment.max_seq_length)

    train_dataloader = DataLoader(train_dataset, batch_size=params.experiment.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=params.experiment.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    metrics_history = []

    for epoch in range(params.experiment.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{params.experiment.num_epochs}")
        train_loss, train_metrics = train_epoch(model, train_dataloader, optimizer, device)
        val_loss, val_metrics = evaluate(model, val_dataloader, device)

        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Train Metrics: {train_metrics}")
        logger.info(f"Val Loss: {val_loss:.4f}")
        logger.info(f"Val Metrics: {val_metrics}")

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        metrics_history.append(epoch_metrics)

    # Convert metrics history to DataFrame and save
    metrics_df = pd.DataFrame(metrics_history)
    metrics_df.set_index("epoch", inplace=True)
    metrics_df.to_csv(Path(params.paths.metrics_dir) / params.data.cls_metrics_file)
    logger.info(f"Metrics history saved to {Path(params.paths.metrics_dir) / params.data.cls_metrics_file}")

    model_dir = Path(params.paths.cls_model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Сохранение токенайзера
    tokenizer.save_pretrained(model_dir)

    # Сохранение модели
    model.base_model.save_pretrained(model_dir)

    torch.save(model.classifier.state_dict(), model_dir / "classifier_head.pth")

    logger.success(f"Model and tokenizer saved in: {model_dir}")


if __name__ == "__main__":
    app()
