import yaml
from dataclasses import dataclass
from marshmallow_dataclass import class_schema
from pathlib import Path

# Определяем параметры для путей
@dataclass
class Paths:
    project_root: str
    data_dir: str
    raw_data_dir: str
    interim_data_dir: str
    processed_data_dir: str
    external_data_dir: str
    models_dir: str
    reports_dir: str
    figures_dir: str

# Определяем параметры для данных
@dataclass
class DataParams:
    train_file: str
    test_file: str
    sample_submission_file: str
    test_labels_file: str
    train_split: float
    random_state: int

# Определяем параметры для экспериментов
@dataclass
class ExperimentParams:
    model_architectures: list
    batch_size: int
    learning_rate: float
    num_epochs: int
    max_seq_length: int
    tokenizer_name: str
    seed: int

# Определяем параметры для модели
@dataclass
class ModelParams:
    pretrained_model_name: str
    hidden_size: int
    num_classes: int

# Определяем параметры для логирования
@dataclass
class LoggingParams:
    experiment_tracker: str
    log_dir: str
    save_logs_to_repo: bool

# Основной класс для всех параметров пайплайна
@dataclass
class PipelineParams:
    paths: Paths
    data: DataParams
    experiment: ExperimentParams
    model: ModelParams
    logging: LoggingParams

# Создаем схему для валидации и загрузки параметров
PipelineParamsSchema = class_schema(PipelineParams)

def read_pipeline_params(path: str) -> PipelineParams:
    """
    Читает параметры из YAML файла и валидирует их.

    :param path: путь к YAML файлу с параметрами
    :return: объект PipelineParams с загруженными параметрами
    """
    with open(path, "r") as input_stream:
        schema = PipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))