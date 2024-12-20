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
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
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

log_file = log_dir / "train_special_clsx.log"
logger.add(log_file, rotation="10 MB", retention="10 days", level="INFO")

writer = SummaryWriter(log_dir="runs")

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

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1),
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_embeddings = last_hidden_state[:, 1 : 1 + self.num_classes, :]
        logits = self.classifier(cls_embeddings).squeeze(-1)
        return logits


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

    num_classes = train_labels.shape[1]

    special_tokens = [f"[CLS_{i}]" for i in range(1, num_classes + 1)]
    tokenizer.add_tokens(special_tokens)

    base_model = AutoModel.from_pretrained(params.model.pretrained_model_name)
    base_model.resize_token_embeddings(len(tokenizer))

    model = SpecialCLSClassifier(base_model, num_classes, tokenizer)
    optimizer = AdamW(model.parameters(), lr=params.experiment.learning_rate)

    total_steps = len(train_texts) // params.experiment.batch_size * params.experiment.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

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

    metrics_df.to_csv(Path(params.paths.metrics_dir) / params.data.cls_x_metrics_file)
    logger.info(f"Metrics history saved to {Path(params.paths.metrics_dir) / params.data.cls_x_metrics_file}")

    model_dir = Path(params.paths.clsx_model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    tokenizer.save_pretrained(model_dir)

    model.base_model.save_pretrained(model_dir)

    torch.save(model.classifier.state_dict(), model_dir / "classifier_head.pth")

    logger.success(f"Model and tokenizer saved to: {model_dir}")


if __name__ == "__main__":
    app()
