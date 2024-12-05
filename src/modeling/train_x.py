import sys
from pathlib import Path
from transformers import AutoModel, AutoTokenizer, AdamW
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score
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

log_file = log_dir / "train_clsx.log"
logger.add(log_file, rotation="10 MB", retention="10 days", level="INFO")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True  # Гарантирует детерминизм в CUDNN
    torch.backends.cudnn.benchmark = False  # Отключает динамическую оптимизацию


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

        # Получаем ID для специального токена '[CLS_X]'
        self.cls_x_token = "[CLS_X]"
        self.tokenizer.add_special_tokens({"additional_special_tokens": [self.cls_x_token]})
        self.cls_x_token_id = self.tokenizer.convert_tokens_to_ids(self.cls_x_token)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        # Добавляем '[CLS_X]' в начало текста
        text_with_cls_x = f"{self.cls_x_token} {text}"
        encoded = self.tokenizer(
            text_with_cls_x,
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
        # Извлекаем выходы для токена '[CLS_X]'
        cls_x_output = outputs.last_hidden_state[
            :, 1, :
        ]  # Предполагаем, что '[CLS_X]' на позиции 1
        return self.classifier(cls_x_output)


def train_epoch(model, dataloader, optimizer, device):
    """
    Trains the model for one epoch.

    Args:
        model: The model to train.
        dataloader: DataLoader for training data.
        optimizer: Optimizer for updating model weights.
        device: Device for computation (CPU/GPU).

    Returns:
        tuple: Average loss and F1 score for the epoch.
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
        preds = torch.sigmoid(outputs).cpu().detach().numpy() > 0.5
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = np.mean(losses)
    f1 = f1_score(all_labels, all_preds, average="micro")
    return avg_loss, f1


def evaluate(model, dataloader, device):
    """
    Evaluates the model on a validation dataset.

    Args:
        model: The model to evaluate.
        dataloader: DataLoader for validation data.
        device: Device for computation (CPU/GPU).

    Returns:
        tuple: Average loss and F1 score for the validation dataset.
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

            preds = torch.sigmoid(outputs).cpu().detach().numpy() > 0.5
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = np.mean(losses)
    f1 = f1_score(all_labels, all_preds, average="micro")
    return avg_loss, f1


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
    # Добавляем специальный токен '[CLS_X]'
    tokenizer.add_special_tokens({"additional_special_tokens": ["[CLS_X]"]})

    base_model = AutoModel.from_pretrained(params.model.pretrained_model_name)
    # Обновляем эмбеддинги модели для нового токена
    base_model.resize_token_embeddings(len(tokenizer))

    # Инициализация модели и оптимизатора
    num_classes = train_labels.shape[1]  # Количество меток
    model = CLSClassifier(base_model, num_classes)
    optimizer = AdamW(model.parameters(), lr=params.experiment.learning_rate)

    # Подготовка данных
    train_dataset = CustomDataset(train_texts, train_labels, tokenizer)
    val_dataset = CustomDataset(val_texts, val_labels, tokenizer)

    train_dataloader = DataLoader(
        train_dataset, batch_size=params.experiment.batch_size, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=params.experiment.batch_size, shuffle=False
    )

    # Основной цикл обучения
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(params.experiment.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{params.experiment.num_epochs}")
        train_loss, train_f1 = train_epoch(model, train_dataloader, optimizer, device)
        val_loss, val_f1 = evaluate(model, val_dataloader, device)
        logger.info(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")

    model_dir = Path(params.paths.clsx_model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Сохранение токенайзера
    tokenizer.save_pretrained(model_dir)

    # Сохранение модели
    model.base_model.save_pretrained(model_dir)

    torch.save(model.classifier.state_dict(), model_dir / "classifier_head.pth")

    logger.success(f"Модель и токенайзер сохранены в: {model_dir}")


if __name__ == "__main__":
    app()
