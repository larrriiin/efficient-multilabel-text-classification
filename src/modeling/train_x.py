import pandas as pd
from pathlib import Path
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AdamW
from sklearn.metrics import f1_score
import numpy as np
import yaml
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

project_root = Path("your_project_path")  # Укажите корень вашего проекта

# Настройка TensorBoard и модели
log_dir = project_root / ".." / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(log_dir=str(log_dir))

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

num_epochs = params["experiment"]["num_epochs"]
batch_size = params["experiment"]["batch_size"]
learning_rate = params["experiment"]["learning_rate"]
log_dir = params["logging"]["log_dir"]

# Пути к данным

train_path = project_root / ".." / "data" / "processed" / "train.csv"
val_path = project_root / ".." / "data" / "processed" / "val.csv"
model_dir = project_root / ".." / "models"

# Чтение данных
train_data = pd.read_csv(train_path)
val_data = pd.read_csv(val_path)

train_data = train_data.dropna(subset=["comment_text"])
val_data = val_data.dropna(subset=["comment_text"])

# Преобразуем в строки, если необходимо
train_data["comment_text"] = train_data["comment_text"].astype(str)
val_data["comment_text"] = val_data["comment_text"].astype(str)

# Выбор текста и меток
train_texts = train_data["comment_text"].tolist()
train_labels = train_data[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values

val_texts = val_data["comment_text"].tolist()
val_labels = val_data[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values

# 1. Загрузка модели и токенайзера
MODEL_NAME = params["experiment"]["tokenizer_name"]

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
base_model = AutoModel.from_pretrained(MODEL_NAME)

# Добавление специальных токенов
special_tokens = [f"[CLS_{i}]" for i in range(1, params["model"]["num_classes"] + 1)]
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
base_model.resize_token_embeddings(len(tokenizer))


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
            text + " " + " ".join(special_tokens),  # Добавляем [CLS_X] токены к тексту
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


# 3. Классификатор с [CLS_X]-токенами
class CLSXClassifier(nn.Module):
    def __init__(self, base_model, num_classes):
        super(CLSXClassifier, self).__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(base_model.config.hidden_size, base_model.config.hidden_size),
                nn.Tanh(),
                nn.Linear(base_model.config.hidden_size, 1),
            )
            for _ in range(num_classes)
        ])

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeddings = outputs.last_hidden_state[:, 1:1 + self.num_classes, :]  # [CLS_1], ..., [CLS_6]

        logits = torch.cat([
            classifier(cls_embeddings[:, i, :]).unsqueeze(1) for i, classifier in enumerate(self.classifiers)
        ], dim=1)

        return logits


# 4. Инициализация модели и оптимизатора
num_classes = train_labels.shape[1]  # Количество меток
model = CLSXClassifier(base_model, num_classes)
model.base_model.resize_token_embeddings(len(tokenizer))  # Увеличиваем эмбеддинги для дополнительных токенов
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

# Сохранение лучшей модели
best_val_f1 = 0.0  # Для отслеживания лучшей метрики
for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")

    # Обучение
    train_loss, train_f1 = train_epoch(model, train_dataloader, optimizer, device)
    writer.add_scalar("Loss/train", train_loss, epoch + 1)
    writer.add_scalar("F1/train", train_f1, epoch + 1)
    print(f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")

    # Валидация
    val_loss, val_f1 = evaluate(model, val_dataloader, device)
    writer.add_scalar("Loss/val", val_loss, epoch + 1)
    writer.add_scalar("F1/val", val_f1, epoch + 1)
    print(f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")

    # Сохранение лучшей модели
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        model_save_path = model_dir / f"best_model_epoch_{epoch + 1}.pth"
        torch.save(model.state_dict(), model_save_path)
        print(f"Лучшее значение F1: {best_val_f1:.4f}. Модель сохранена в {model_save_path}.")

# Закрытие TensorBoard
writer.close()

# Сохранение токенайзера
tokenizer.save_pretrained(model_dir)

# Сохранение модели
model.base_model.save_pretrained(model_dir)

torch.save(model.classifier.state_dict(), model_dir / "classifier_head_x.pth")

print(f"Модель и токенайзер сохранены в: {model_dir}")
