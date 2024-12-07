"""
A script for training and evaluating a multilabel text
classification model using PyTorch and Transformers with special CLS tokens and additional metrics.

This script performs the following tasks:
1. Reads pipeline parameters and data paths.
2. Prepares and tokenizes the training and validation datasets.
3. Adds special tokens [CLS_1], [CLS_2], ..., [CLS_num_classes] to the tokenizer and input texts.
4. Initializes a transformer-based classification model with a custom head for each special CLS token.
5. Trains the model over multiple epochs, logging performance metrics (loss, precision, recall, F1 scores).
6. Evaluates the model on the validation dataset.
7. Saves the trained model and tokenizer for future use.
8. Generates a metrics table and saves it as a CSV file.
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

log_file = log_dir / "train_special_clsx.log"
logger.add(log_file, rotation="10 MB", retention="10 days", level="INFO")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
    torch.backends.cudnn.benchmark = False  # Disable dynamic optimization


class CustomDataset(Dataset):
    """
    A Dataset for tokenizing text data with labels and adding special CLS tokens.

    Args:
        texts (list[str]): Input texts.
        labels (list): Labels for the texts.
        tokenizer: Tokenizer for text processing.
        num_classes (int): Number of output classes.
        max_length (int): Max sequence length (default: 128).
    """

    def __init__(self, texts, labels, tokenizer, num_classes, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.num_classes = num_classes
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        # Add special CLS tokens after the model's starting token. They will appear after <s>.
        special_cls_tokens = [f"[CLS_{i}]" for i in range(1, self.num_classes + 1)]
        text_with_special_tokens = " ".join(special_cls_tokens) + " " + text
        
        encoded = self.tokenizer(
            text_with_special_tokens,
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


class SpecialCLSClassifier(nn.Module):
    """
    A classification model that uses special CLS tokens for each class.
    """

    def __init__(self, base_model, num_classes, tokenizer):
        super(SpecialCLSClassifier, self).__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.hidden_size = base_model.config.hidden_size

        # One classifier head to be applied to each special CLS token embedding
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        # Positions: <s> at index 0, then [CLS_1] at 1, [CLS_2] at 2, ..., [CLS_num_classes] at num_classes.
        # So we can directly slice these embeddings:
        cls_embeddings = last_hidden_state[:, 1 : 1 + self.num_classes, :]
        logits = self.classifier(cls_embeddings).squeeze(-1)  # [batch_size, num_classes]
        return logits


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

    num_classes = train_labels.shape[1]  # Number of labels

    # Add special CLS tokens to the tokenizer
    special_tokens = [f"[CLS_{i}]" for i in range(1, num_classes + 1)]
    tokenizer.add_tokens(special_tokens)

    base_model = AutoModel.from_pretrained(params.model.pretrained_model_name)
    base_model.resize_token_embeddings(len(tokenizer))

    model = SpecialCLSClassifier(base_model, num_classes, tokenizer)
    optimizer = AdamW(model.parameters(), lr=params.experiment.learning_rate)

    train_dataset = CustomDataset(train_texts, train_labels, tokenizer, num_classes, max_length=params.experiment.max_seq_length)
    val_dataset = CustomDataset(val_texts, val_labels, tokenizer, num_classes, max_length=params.experiment.max_seq_length)

    train_dataloader = DataLoader(
        train_dataset, batch_size=params.experiment.batch_size, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=params.experiment.batch_size, shuffle=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Initialize metrics storage
    metrics_history = []

    for epoch in range(params.experiment.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{params.experiment.num_epochs}")
        train_loss, train_metrics = train_epoch(model, train_dataloader, optimizer, device)
        val_loss, val_metrics = evaluate(model, val_dataloader, device)

        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Train Metrics: {train_metrics}")
        logger.info(f"Val Loss: {val_loss:.4f}")
        logger.info(f"Val Metrics: {val_metrics}")

        # Store metrics
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        metrics_history.append(epoch_metrics)

    # Convert metrics history to DataFrame
    metrics_df = pd.DataFrame(metrics_history)
    metrics_df.set_index("epoch", inplace=True)

    # Save metrics table to CSV
    metrics_df.to_csv("metrics_history.csv")
    logger.info("Metrics history saved to metrics_history.csv")

    model_dir = Path(params.paths.clsx_model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save tokenizer
    tokenizer.save_pretrained(model_dir)

    # Save model
    model.base_model.save_pretrained(model_dir)

    # Save classifier state dict
    torch.save(model.classifier.state_dict(), model_dir / "classifier_head.pth")

    logger.success(f"Model and tokenizer saved to: {model_dir}")


if __name__ == "__main__":
    app()
