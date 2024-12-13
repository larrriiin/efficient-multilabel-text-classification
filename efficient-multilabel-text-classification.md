# efficient-multilabel-text-classification
## _Alexey Larin, Yuriy Vinnik, Ivan Lisitsyn, Max Goloviznin_

[![late 2024]()](https://github.com/larrriiin/efficient-multilabel-text-classification)

## Decription
A project exploring efficient transformer-based architectures for multilabel text classification using the Jigsaw Toxic Comment Classification dataset. 

## Includes 
- data preprocessing, 
- model implementation, 
- experimental comparison, 
- detailed results analysis. 

## Commands
The [![ Makefile]()](https://github.com/larrriiin/efficient-multilabel-text-classification/blob/main/Makefile) contains the central entry points for common tasks related to this project.

## Data preprocessing
```
def preprocess_text(text: str, remove_digits=True) -> str:
    if remove_digits:
        text = re.sub(r"[^a-zA-Z\s]", "", text)
    else:
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    # Удаление символов, кроме букв, цифр и пробелов
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    # Приведение к нижнему регистру
    text = text.lower()
    # Удаление лишних пробелов
    text = re.sub(r"\s+", " ", text).strip()
    # Разделение текста на слова
    words = text.split()
    # Удаление стоп-слов и лемматизация
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    # Сборка обратно в строку
    text = " ".join(words)
    return text
```
```
train_data, val_data = train_test_split(df,
    test_size=1 - params.data.train_split,
    random_state=params.data.random_state)
processed_data_dir = Path("../") / Path(params.paths.processed_data_dir)
processed_data_dir.mkdir(parents=True, exist_ok=True)
    
train_data.to_csv(processed_data_dir / "train.csv", index=False)
val_data.to_csv(processed_data_dir / "val.csv", index=False)
logger.info(f"Processed train and validation datasets saved in {processed_data_dir}")
)
INFO     | __main__:<module>:1 - Processed train and validation datasets saved in ..\data\processed
```

## Model implementation
[![train_CLS.py]()](https://github.com/larrriiin/efficient-multilabel-text-classification/blob/main/src/modeling/train_CLS.py)
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







