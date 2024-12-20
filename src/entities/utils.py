import numpy as np
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
import torch.nn as nn
from loguru import logger

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
    try:
        metrics.update({
            f"roc_auc_class_{i}": roc_auc_score(labels[:, i], preds[:, i]) for i in range(labels.shape[1])
        })
    except ValueError:
        logger.warning("ROC-AUC computation failed for some classes.")
    return metrics

def train_epoch(model, dataloader, optimizer, scheduler, device, loss_metric, writer):
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
    all_preds = []
    all_labels = []

    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids, attention_mask)
        loss = nn.BCEWithLogitsLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_metric.update(loss.item())

        preds = (torch.sigmoid(outputs).cpu().detach().numpy() > 0.5).astype(int)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

        if step % 10 == 0:
            writer.add_scalar("Loss/Train", loss.item(), step)

    avg_loss = loss_metric.compute().item()
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
