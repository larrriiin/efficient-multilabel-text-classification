import sys
from pathlib import Path
import typer
import pandas as pd
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from loguru import logger

log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)  # Создаем папку, если её нет

log_file = log_dir / "data_preparation.log"
logger.add(log_file, rotation="10 MB", retention="10 days", level="INFO")

project_root = Path().resolve().parent
sys.path.append(str(project_root / "src"))

from entities.params import read_pipeline_params

app = typer.Typer()

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Инициализация стоп-слов и лемматайзера
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


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


@app.command()
def main(params_path: str):
    params = read_pipeline_params(params_path)

    logger.info("Reading raw data")
    df = pd.read_csv(Path(params.paths.raw_data_dir) / params.data.train_file)
    df.dropna(inplace=True)

    logger.info("Preprocessing text column")
    df['comment_text'] = df['comment_text'].apply(preprocess_text)

    train_data, val_data = train_test_split(
        df,
        test_size=1 - params.data.train_split,
        random_state=params.data.random_state
    )

    processed_data_dir = Path(params.paths.processed_data_dir)
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    train_data.to_csv(processed_data_dir / params.data.train_file, index=False)
    val_data.to_csv(processed_data_dir / params.data.val_file, index=False)

    logger.success(f"Processed train and validation datasets saved in {processed_data_dir}")


if __name__ == "__main__":
    app()
