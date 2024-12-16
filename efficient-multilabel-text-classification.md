# efficient-multilabel-text-classification
### _Alexey Larin, Yuriy Vinnik, Ivan Lisitsyn, Max Goloviznin_

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
The preprocessing step involves cleaning the text data and preparing it for training.
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
After preprocessing, the data is split into training and validation sets and saved as CSV files:
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


[![train_CLS_X.py]()](https://github.com/larrriiin/efficient-multilabel-text-classification/blob/main/src/modeling/train_CLS_X.py)
A script for training and evaluating a multilabel textclassification model using PyTorch and Transformers with special CLS tokens and additional metrics.

This script performs the following tasks:
1. Reads pipeline parameters and data paths.
2. Prepares and tokenizes the training and validation datasets.
3. Adds special tokens [CLS_1], [CLS_2], ..., [CLS_num_classes] to the tokenizer and input texts.
4. Initializes a transformer-based classification model with a custom head for each special CLS token.
5. Trains the model over multiple epochs, logging performance metrics (loss, precision, recall, F1 scores).
6. Evaluates the model on the validation dataset.
7. Saves the trained model and tokenizer for future use.
8. Generates a metrics table and saves it as a CSV file.



## Results Analysys
Для сравнения моделей, необходимо рассмотреть метрики, наиболее важные для вашей задачи мультиклассификации. Обычно это `val_f1_micro` и `val_f1_macro`, так как они дают понимание качества предсказаний на валидационных данных:

1. `val_f1_micro` — учитывает точность и полноту всех классов, делая акцент на сбалансированности.
2. `val_f1_macro` — среднее значение F1 для всех классов, одинаково учитывающее каждый класс.

### Анализ метрик
Из таблиц видно, что:
| epoch | train_loss           | val_loss             | train_precision_micro | train_recall_micro | train_f1_micro     | train_precision_macro | train_recall_macro | train_f1_macro     | val_precision_micro | val_recall_micro   | val_f1_micro       | val_precision_macro | val_recall_macro    | val_f1_macro        |
|-------|----------------------|----------------------|-----------------------|--------------------|--------------------|-----------------------|--------------------|--------------------|---------------------|--------------------|--------------------|---------------------|---------------------|---------------------|
| 1     | 0.06206429489200354  | 0.04530734757870613  | 0.7813236541906289    | 0.5914573222951756 | 0.6732604898655511 | 0.5176717905108884    | 0.3494757554525923 | 0.3930207091100073 | 0.8002233923727461  | 0.7089341249646594 | 0.7518177048197288 | 0.5011965185691741  | 0.41060172648290844 | 0.42506755788470657 |
| 2     | 0.043320849383198155 | 0.04283728983765585  | 0.8228427461683813    | 0.6992577790465315 | 0.7560331024904029 | 0.708985377944582     | 0.4702537699440181 | 0.5297437999289965 | 0.8367101648351648  | 0.6888606163415324 | 0.7556210265157389 | 0.7080262144378371  | 0.5395161610561768  | 0.6059547479652555  |
| 3     | 0.038258083818435394 | 0.043145096231380915 | 0.8310403089548636    | 0.737153868113046  | 0.7812866381755607 | 0.7255200822884135    | 0.5476983051901593 | 0.6102356636255281 | 0.7367682693533988  | 0.8166525303929885 | 0.7746563861884009 | 0.5930141253392877  | 0.703335120354045   | 0.6413652677068505  |

| epoch | train_loss           | val_loss             | train_precision_micro | train_recall_micro | train_f1_micro     | train_precision_macro | train_recall_macro | train_f1_macro      | val_precision_micro | val_recall_micro   | val_f1_micro       | val_precision_macro | val_recall_macro    | val_f1_macro       |
|-------|----------------------|----------------------|-----------------------|--------------------|--------------------|-----------------------|--------------------|---------------------|---------------------|--------------------|--------------------|---------------------|---------------------|--------------------|
| 1     | 0.06302905762416997  | 0.04573365581360614  | 0.8034369970406521    | 0.5522052526405937 | 0.6545416093898698 | 0.5902014495638673    | 0.3349885038670419 | 0.39939808457339415 | 0.7771030805687204  | 0.7417302798982188 | 0.7590047736149282 | 0.6153063445307314  | 0.47656616820969533 | 0.5035652200001325 |
| 2     | 0.047379076821714494 | 0.042268810980517955 | 0.8181289167412713    | 0.6521909791607193 | 0.7257962036375188 | 0.6222512735171963    | 0.4170567836986013 | 0.47493380167880517 | 0.8106301718650541  | 0.7201017811704835 | 0.7626890253031892 | 0.6301768253130288  | 0.4745374209705278  | 0.5121986938276663 |
| 3     | 0.04165666352402165  | 0.04123250526805113  | 0.8269641287514127    | 0.7050028546959749 | 0.7611287681787537 | 0.7211935667819507    | 0.4936434431822061 | 0.556720042535711   | 0.7870099473376243  | 0.7605315238903025 | 0.7735442127965493 | 0.6638966075035723  | 0.5708239373999434  | 0.60261756247547   |
#### Первая модель:
- Лучшее val_f1_micro на 3 эпохе: 0.7747
- Лучшее val_f1_macro на 3 эпохе: 0.6414
|
#### Вторая модель:
- Лучшее val_f1_micro на 3 эпохе: 0.7735
- Лучшее val_f1_macro на 3 эпохе: 0.6026

### Выводы:
1. `val_f1_micro`: Первая модель немного лучше (0.7747 против 0.7735).
2. `val_f1_macro`: Первая модель значительно лучше (0.6414 против 0.6026).

Это указывает на то, что первая модель лучше справляется с предсказанием всех классов (особенно редких), что важно для мультиклассификации.

Дополнительный фактор:
- Потери на валидации (val_loss) у второй модели ниже, но это не компенсирует разницы в метриках F1.

### Итог:
Первая модель справилась лучше, так как достигла лучших значений F1-метрик (особенно Macro), что критически важно для мультиклассификации.
### Recommendations 
Choose the first model for deployment due to its superior perfomance in F1 metrics.
Consider fine-tuning the model with additional data or exploring other transformet architectures for better results.





