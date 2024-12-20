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
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import pandas as pd
import typer
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
import torchmetrics

sys.path.append(str(Path(__file__).resolve().parent.parent))

from entities.params import read_pipeline_params
from entities.utils import set_seed, train_epoch, evaluate

app = typer.Typer()

project_root = Path().resolve().parent

log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)

log_file = log_dir / "train_cls_extended_metrics.log"
logger.add(log_file, rotation="10 MB", retention="10 days", level="INFO")

writer = SummaryWriter(log_dir="runs")

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

    total_steps = len(train_texts) // params.experiment.batch_size * params.experiment.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    train_dataset = CustomDataset(train_texts, train_labels, tokenizer, max_length=params.experiment.max_seq_length)
    val_dataset = CustomDataset(val_texts, val_labels, tokenizer, max_length=params.experiment.max_seq_length)

    train_dataloader = DataLoader(train_dataset, batch_size=params.experiment.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=params.experiment.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_metric = torchmetrics.MeanMetric().to(device)
    metrics_history = []

    for epoch in range(params.experiment.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{params.experiment.num_epochs}")
        train_loss, train_metrics = train_epoch(model, train_dataloader, optimizer, scheduler, device, loss_metric, writer=writer)
        val_loss, val_metrics = evaluate(model, val_dataloader, device)

        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Train Metrics: {train_metrics}")
        logger.info(f"Val Loss: {val_loss:.4f}")
        logger.info(f"Val Metrics: {val_metrics}")

        writer.add_scalar("Loss/Validation", val_loss, epoch)
        for metric, value in train_metrics.items():
            writer.add_scalar(f"Metrics/Train/{metric}", value, epoch)
        for metric, value in val_metrics.items():
            writer.add_scalar(f"Metrics/Validation/{metric}", value, epoch)

        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        metrics_history.append(epoch_metrics)

    metrics_df = pd.DataFrame(metrics_history)
    metrics_df.set_index("epoch", inplace=True)
    metrics_df.to_csv(Path(params.paths.metrics_dir) / params.data.cls_metrics_file)
    logger.info(f"Metrics history saved to {Path(params.paths.metrics_dir) / params.data.cls_metrics_file}")

    model_dir = Path(params.paths.cls_model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    tokenizer.save_pretrained(model_dir)

    model.base_model.save_pretrained(model_dir)

    torch.save(model.classifier.state_dict(), model_dir / "classifier_head.pth")

    logger.success(f"Model and tokenizer saved in: {model_dir}")


if __name__ == "__main__":
    app()
