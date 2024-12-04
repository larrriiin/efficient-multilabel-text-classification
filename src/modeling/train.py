from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW
from sklearn.metrics import f1_score
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pandas as pd

# 1. Загрузка модели и токенайзера
MODEL_NAME = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModel.from_pretrained(MODEL_NAME)

project_root = Path().resolve().parent

train_path = project_root / ".." / "data" / "processed" / "train.csv"
val_path = project_root / ".." / "data" / "processed" / "val.csv"
model_dir = project_root / ".." / "models"

train_data = pd.read_csv(train_path)
val_data = pd.read_csv(val_path)

# Выбор текста и меток
train_texts = train_data["comment_text"].tolist()
train_labels = train_data[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values

val_texts = val_data["comment_text"].tolist()
val_labels = val_data[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values

# 1. Загрузка модели и токенайзера
MODEL_NAME = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModel.from_pretrained(MODEL_NAME)


# 2. Датасет
class CustomDataset(Dataset):
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


# 3. Классификатор с [CLS]-токеном
class CLSClassifier(nn.Module):
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


# 4. Инициализация модели и оптимизатора
num_classes = train_labels.shape[1]  # Количество меток
model = CLSClassifier(base_model, num_classes)
optimizer = AdamW(model.parameters(), lr=1e-5)


# 5. Тренировочный цикл
def train_epoch(model, dataloader, optimizer, device):
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


# 6. Валидация
def evaluate(model, dataloader, device):
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


# 7. Подготовка данных
train_dataset = CustomDataset(train_texts, train_labels, tokenizer)
val_dataset = CustomDataset(val_texts, val_labels, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# 8. Основной цикл
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

num_epochs = 3
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train_loss, train_f1 = train_epoch(model, train_dataloader, optimizer, device)
    val_loss, val_f1 = evaluate(model, val_dataloader, device)
    print(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")


# Сохранение токенайзера
tokenizer.save_pretrained(model_dir)

# Сохранение модели
model.base_model.save_pretrained(model_dir)

torch.save(model.classifier.state_dict(), model_dir / "classifier_head.pth")

print(f"Модель и токенайзер сохранены в: {model_dir}")
