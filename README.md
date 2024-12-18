# Эффективный MultiLabel классификатор
### _Алексей Ларин, Юрий Винник, Иван Лисицин, Максим Головизнин_

## Структура проекта (ccds)
```
├── Makefile           <- Makefile
├── README.md          <- Основной README-файл.
├── data  
│   ├── external       <- Данные из сторонних источников.  
│   ├── interim        <- Промежуточные данные, которые были преобразованы.  
│   ├── processed      <- Финальные, канонические наборы данных для моделирования.  
│   └── raw            <- Исходные, неизменные данные.  
│  
├── docs               <- Стандартный проект на основе mkdocs; подробности на www.mkdocs.org  
│  
├── models             <- Обученные и сериализованные модели, предсказания моделей или их сводки  
│  
├── notebooks          <- Jupyter-ноутбуки
│  
├── params.yaml        <- Конфигурационный файл
│  
├── pyproject.toml     <- Конфигурационный файл проекта с метаданными пакета для  
│                         исходного кода и конфигурацией для инструментов, таких как black  
│  
├── reports  
│   └── metrics        <- Метрики, полученные в процессе обучения моделей  
│  
├── requirements.txt   <- Файл с зависимостями для воспроизведения окружения анализа
│  
├── setup.cfg          <- Конфигурационный файл для flake8  
│  
└── src   <- Исходный код, используемый в этом проекте.  
    │  
    ├── __init__.py                 <- Делает src Python-модулем  
    │  
    ├── dataset_preprocessing.py    <- Скрипты для предобработки данных  
    │  
    ├── modeling                
    │   ├── __init__.py 
    │   ├── train_CLS.py              <- Код для обучения модели CLS  
    │   └── train_CLS_X.py            <- Код для обучения модели CLS_X  
    └── entities                
        ├── __init__.py 
        └── params.py                 <- Код для создания класса параметров  

```

## Описание
[Проект](https://github.com/larrriiin/efficient-multilabel-text-classification), исследующий эффективные архитектуры на основе трансформеров для многомаркировочной классификации текста с использованием набора данных Jigsaw Toxic Comment Classification.

## Определение проблемы
Обсуждение важных для вас тем в интернете может быть очень сложной задачей. Риск оскорблений и преследования часто отпугивает людей от выражения своего мнения и заставляет их воздерживаться от участия в дискуссиях с противоположными точками зрения. В результате многим онлайн-платформам трудно создать здоровую атмосферу для обсуждений, что вынуждает некоторые сообщества ограничивать или даже полностью отключать комментарии пользователей. Это не только подавляет диалог, но и ограничивает вовлечённость сообщества.

Чтобы решить эти проблемы, команда Conversation AI — инициатива, созданная Jigsaw и Google — работает над инструментами, направленными на улучшение онлайн-коммуникации. Одним из ключевых направлений их работы является понимание негативного поведения в интернете, особенно токсичных комментариев. Токсичность может включать грубые, неуважительные и вредоносные высказывания, которые могут отпугнуть пользователей от участия в обсуждениях. Хотя через Perspective API они уже разработали несколько общедоступных моделей для оценки токсичности, остаются определённые пробелы. Эти модели могут допускать ошибки и не позволяют пользователям указывать, какие типы токсичных комментариев они хотят отслеживать — например, различать нецензурную лексику и другие вредоносные высказывания.

В этом проекте мы беремся за задачу создания многоцелевой модели, которая будет обнаруживать различные виды токсичности, такие как угрозы, нецензурная лексика, оскорбления и ненависть на основе идентичности. Наш подход использует набор данных с комментариями из правок на страницах обсуждений в Википедии, который представляет собой богатый источник пользовательских взаимодействий.

В рамках проекта мы реализовали несколько эффективных архитектур на основе трансформеров для многомаркировочной классификации текста. Эти современные модели могут анализировать комментарии всесторонне, позволяя выявлять несколько типов токсичности одновременно. Мы считаем, что улучшение точности и детализации в идентификации токсичных комментариев предоставит платформам более качественные инструменты для поощрения уважительного взаимодействия между пользователями.

## Проект включает в себя
- предобработку данных;
- реализацию моделей;
- экспериментальное сравнение, 
- анализ результатов. 

## Предобработка данных
Этап предобработки включает очистку текстовых данных и их подготовку для обучения.
```py
def preprocess_text(text: str, remove_digits=True) -> str:
    if remove_digits:
        text = re.sub(r"[^a-zA-Z\s]", "", text)
    else:
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    text = " ".join(words)
    return text
```
После предобработки данные разделяются на обучающий и валидационный наборы и сохраняются в виде CSV-файлов:
```py
train_data, val_data = train_test_split(
    df,
    test_size=1 - params.data.train_split,
    random_state=params.data.random_state
)

processed_data_dir = Path("../") / Path(params.paths.processed_data_dir)
processed_data_dir.mkdir(parents=True, exist_ok=True)

train_data.to_csv(processed_data_dir / "train.csv", index=False)
val_data.to_csv(processed_data_dir / "val.csv", index=False)

logger.info(f"Processed train and validation datasets saved in {processed_data_dir}")

```

## Реализация модели
### [train_CLS.py](https://github.com/larrriiin/efficient-multilabel-text-classification/blob/main/src/modeling/train_CLS.py)

Скрипт для обучения и оценки мультилейбл модели классификации текста с использованием PyTorch и Transformers (на основе единственного CLS-встраивания) с расширенными метриками (точность, полнота, F1 - микро и макро).

Этот скрипт выполняет следующие задачи:
1. Считывает параметры пайплайна и пути к данным.
2. Подготавливает и токенизирует обучающие и валидационные наборы данных.
3. Инициализирует модель классификации на основе трансформера с пользовательской головой, использующей встраивание токена [CLS].
4. Обучает модель на протяжении нескольких эпох, логируя показатели производительности (потери, точность, полнота, F1-метрики).
5. Оценивает модель на валидационном наборе данных.
6. Сохраняет обученную модель и токенизатор для дальнейшего использования.
7. Сохраняет таблицу с историей метрик в виде CSV-файла.

Классы:
- **CustomDataset**: Класс набора данных для токенизации текстов и обработки меток.
- **CLSClassifier**: Модель классификации с трансформерным "скелетом" и пользовательской головой классификатора.

Функции:
- **train_epoch**: Обучает модель за одну эпоху и вычисляет средние потери и метрики.
- **evaluate**: Оценивает модель на валидационном наборе данных и вычисляет средние потери и метрики.


### [train_CLS_X.py](https://github.com/larrriiin/efficient-multilabel-text-classification/blob/main/src/modeling/train_CLS_X.py)

Скрипт для обучения и оценки мультилейбл модели классификации текста с использованием PyTorch и Transformers, со специальными CLS токенами и дополнительными метриками.

Этот скрипт выполняет следующие задачи:
1. Считывает параметры пайплайна и пути к данным.  
2. Подготавливает и токенизирует обучающие и валидационные наборы данных.  
3. Добавляет специальные токены **[CLS_1], [CLS_2], ..., [CLS_num_classes]** в токенизатор и текстовые входные данные.  
4. Инициализирует модель классификации на основе трансформера с пользовательской головой для каждого специального **CLS** токена.  
5. Обучает модель на протяжении нескольких эпох, логируя показатели производительности (потери, точность, полнота, F1-метрики).  
6. Оценивает модель на валидационном наборе данных.  
7. Сохраняет обученную модель и токенизатор для дальнейшего использования.  
8. Генерирует таблицу метрик и сохраняет её в виде **CSV-файла**.  

Как запустить:
Запустите скрипт с помощью `typer`:
```shell
python train.py --params-path <path_to_params_file>
```

## Анализ результатов
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
#### Первая модель (CLS):
- Лучшее val_f1_micro на 3 эпохе: 0.7747
- Лучшее val_f1_macro на 3 эпохе: 0.6414

#### Вторая модель (CLS_X):
- Лучшее val_f1_micro на 3 эпохе: 0.7735
- Лучшее val_f1_macro на 3 эпохе: 0.6026

### Выводы:
1. `val_f1_micro`: Первая модель (CLS) немного лучше (0.7747 против 0.7735).
2. `val_f1_macro`: Первая модель (CLS) значительно лучше (0.6414 против 0.6026).

Это указывает на то, что первая модель (CLS) лучше справляется с предсказанием всех классов (особенно редких), что важно для мультиклассификации.

Дополнительный фактор:
- Потери на валидации (val_loss) у второй модели ниже, но это не компенсирует разницы в метриках F1.

### Итог:
Первая модель справилась лучше, так как достигла лучших значений F1-метрик (особенно Macro), что критически важно для мультиклассификации.

## Альтернативные подходы  
Помимо предложенных подходов (использование **CLS**-встраивания и добавление новых **CLS_X** токенов), можно рассмотреть несколько "дополненных" архитектур для многомаркировочной классификации. Вот некоторые из них:  

### 1. Среднее/Взвешенное среднее всех токенов  
- Вместо использования только **CLS** или новых добавленных токенов можно использовать встраивания всех токенов на выходе трансформера.  
- Затем можно вычислить среднее или взвешенное среднее (например, с использованием обучаемых весов для каждого токена).  

### 2. Само-внимание для токенов  
- Дополнительный слой само-внимания можно применить к встраиваниям всех токенов.  
- Это позволит модели сосредоточиться на наиболее значимых токенах для классификации.  

### 3. Комбинирование нескольких встраиваний (**CLS**, начальные и финальные токены)  
- Вместо использования одного **CLS** можно применить несколько встраиваний:  
  - **CLS**  
  - Первый токен (полезен для заголовков или ключевых слов).  
  - Последний токен (который может содержать итоговую информацию текста).  
- Эти встраивания можно объединить (например, через конкатенацию или суммирование).  

### 4. **CNN** поверх встраиваний токенов  
- Применение сверточных нейронных сетей (**CNN**) поверх последовательности токенов.  
- Этот метод подчеркивает локальные зависимости между токенами.  

### 5. Рекуррентные сети (**LSTM/GRU**) поверх встраиваний  
- Добавление слоя **LSTM** или **GRU** к встраиваниям для учета последовательной информации.  

### 6. Многоцелевые классификаторы  
- Вместо одного классификатора можно разработать отдельные классификаторы для каждого класса.  
- Это позволяет учесть особенности каждой метки.  

### 7. Аугментация признаков  
- Комбинирование встраиваний из различных слоев трансформера (например, с помощью обучаемых весов).  
- Такой подход позволяет использовать информацию как из начальных, так и из финальных слоев.  

### 8. Графовое пуллинг-метод  
- Применение методов на основе графов (например, **Graph Attention Networks**, **GAT**) к встраиваниям токенов.  
- Каждый токен рассматривается как узел графа, а связи между токенами определяются на основе внимания.  

### Как выбрать подход?  
- **Когда данных мало**: Используйте архитектуры с меньшим количеством параметров, такие как **CLS** или среднее по токенам.  
- **Когда данных много**: Изучите более сложные архитектуры, такие как само-внимание или многоцелевые классификаторы.  
- **Для длинных текстов**: Рекомендуются подходы, учитывающие информацию со всех токенов (например, **LSTM** или **CNN**).  

### Почему выбран **CLS_X**?  
Ввиду ограниченного размера нашего набора данных использование сложных архитектур не является эффективным подходом. **CLS_X** предоставляет баланс между простотой и способностью модели обрабатывать несколько классов с высокой точностью.  

## Источники

### Статья:

    Text Classification Algorithms: A Survey

### Книги:

    Natural Language Processing in Action
    Speech and Language Processing

### Блоги/Статьи/Ноутбуки:
    
    https://www.kaggle.com/abhinand05/bert-for-humans-tutorial-baseline-version-2
    https://www.kaggle.com/amar09/text-pre-processing-and-feature-extraction
    https://www.kaggle.com/datafan07/disaster-tweets-nlp-eda-bert-with-transformers
    https://www.kaggle.com/gunesevitan/nlp-with-disaster-tweets-eda-cleaning-and-bert
    https://yashuseth.blog/2019/06/12/bert-explained-faqs-understand-bert-working/
    https://gist.github.com/MrEliptik/b3f16179aa2f530781ef8ca9a16499af
    https://machinelearningmastery.com/gentle-introduction-bag-words-model/
