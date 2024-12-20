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
├── runs               <- Логи TensorBoard 
│   │  
│   ├── cls_standard
│   └── cls_special_tokens
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
        ├── params.py                 <- Код для создания класса параметров  
        └── utils.py                  <- Функции для работы кода

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
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
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
8. Логирует потери и метрики (F1, точность, полноту, ROC-AUC) на каждой n-й итерации и после каждой эпохи в TensorBoard для визуализации экспериментов.
9. Использует Learning Rate Scheduler с warmup для улучшения процесса обучения и динамической адаптации скорости обучения.
10. Вычисляет ROC-AUC для каждого класса, чтобы оценить предсказательную способность модели вне зависимости от порога классификации.
11. Применяет torchmetrics.MeanMetric для вычисления среднего значения потерь по эпохе без хранения всех значений в памяти.

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
9. Логирует потери и метрики (F1, точность, полноту, ROC-AUC) на каждой n-й итерации и после каждой эпохи в TensorBoard для визуализации экспериментов.
10. Использует Learning Rate Scheduler с warmup для улучшения процесса обучения и динамической адаптации скорости обучения.
11. Вычисляет ROC-AUC для каждого класса, чтобы оценить предсказательную способность модели вне зависимости от порога классификации.
12. Применяет torchmetrics.MeanMetric для вычисления среднего значения потерь по эпохе без хранения всех значений в памяти.

Как запустить:
Запустите скрипт с помощью `typer`:
```shell
python train.py --params-path <path_to_params_file>
```

## Анализ результатов
Для сравнения моделей, необходимо рассмотреть метрики, наиболее важные для вашей задачи мультиклассификации. Обычно это `val_f1_micro` и `val_f1_macro`, так как они дают понимание качества предсказаний на валидационных данных:

1. `val_f1_micro` — учитывает точность и полноту всех классов, делая акцент на сбалансированности.
2. `val_f1_macro` — среднее значение F1 для всех классов, одинаково учитывающее каждый класс.

### Просмотр логов в TensorBoard
1. Запустите TensorBoard, указав директорию с логами:
```shell
tensorboard --logdir runs
```
2. Откройте указанный адрес (обычно `http://localhost:6006`) в браузере.
3. Логи находятся в поддиректориях:
- `runs/cls_standard` – для модели CLS.
- `runs/cls_special_tokens` – для модели CLS_X.

#### В TensorBoard вы сможете увидеть графики:
- Потери (Train и Validation).
- Метрики (Precision, Recall, F1, ROC-AUC).
- Динамику обучения по эпохам и итерациям.

### Обновленный анализ метрик

Из таблиц видно, что:

#### Таблица метрик CLS:
| epoch | train_loss           | val_loss             | train_precision_micro | train_recall_micro | train_f1_micro     | train_precision_macro | train_recall_macro | train_f1_macro     | val_precision_micro | val_recall_micro   | val_f1_micro       | val_precision_macro | val_recall_macro    | val_f1_macro        | val_roc_auc_class_0 | val_roc_auc_class_1 | val_roc_auc_class_2 | val_roc_auc_class_3 | val_roc_auc_class_4 | val_roc_auc_class_5 |
|-------|----------------------|----------------------|-----------------------|--------------------|--------------------|-----------------------|--------------------|--------------------|---------------------|--------------------|--------------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------|
| 1     | 0.08505300432443619  | 0.042920302955099944 | 0.5684565443523799    | 0.5655152726234656 | 0.5669820939842226 | 0.3977655990769431    | 0.3372221624084885 | 0.34421272143856485 | 0.8081845473218517  | 0.7230703986429178 | 0.7632619562784452 | 0.6631170578116178  | 0.4283306942567635  | 0.44711189338302076 | 0.8864020422730559  | 0.5958452223962237  | 0.9221605236228836  | 0.5                 | 0.8585256857190845  | 0.5017006802721088  |
| 2     | 0.06287867575883865  | 0.03880636188149671  | 0.8300709918338873    | 0.7218098772480731 | 0.7721642204111238 | 0.7476063111191859    | 0.46779042965866197 | 0.5183344140658099 | 0.8087537091988131  | 0.770568278201866  | 0.7891993629651078 | 0.7117434005184388  | 0.5894570027097958  | 0.632057311850127   | 0.8899804342979015  | 0.6808503211599979  | 0.9344817349834921  | 0.634946698842305   | 0.8799708864880564  | 0.7266736111604124  |
| 3     | 0.053347062319517136 | 0.03867136592294596  | 0.8428331445091222    | 0.7698401370254068 | 0.8046847317282407 | 0.7504126390016199    | 0.5803663449027404 | 0.6412816892828642 | 0.7847820049986115  | 0.7989821882951654 | 0.791818436536845  | 0.6636371250688092  | 0.669466220431124   | 0.6658899234327795  | 0.9040162992789096  | 0.7204150419348916  | 0.9263479620798177  | 0.7629482046350234  | 0.898770604247585   | 0.7701481212326177  |

#### Таблица метрик CLS_X:
| epoch | train_loss           | val_loss             | train_precision_micro | train_recall_micro | train_f1_micro     | train_precision_macro | train_recall_macro | train_f1_macro     | val_precision_micro | val_recall_micro   | val_f1_micro       | val_precision_macro | val_recall_macro    | val_f1_macro        | val_roc_auc_class_0 | val_roc_auc_class_1 | val_roc_auc_class_2 | val_roc_auc_class_3 | val_roc_auc_class_4 | val_roc_auc_class_5 |
|-------|----------------------|----------------------|-----------------------|--------------------|--------------------|-----------------------|--------------------|--------------------|---------------------|--------------------|--------------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------|---------------------|
| 1     | 0.06963680684566498  | 0.03976174565010242  | 0.8135602233448551    | 0.5459249214958607 | 0.6533985350957741 | 0.6140396686498014    | 0.33618917390388026 | 0.4097658304207281 | 0.8299077520634407  | 0.7249081142210914 | 0.7738625216932016 | 0.7974219072034147  | 0.4834007554057456  | 0.5322968748100084  | 0.8797248195278397  | 0.5851000535020155  | 0.9082318073870985  | 0.5067567567567568  | 0.8614074903241565  | 0.6913862569157198  |
| 2     | 0.0542495958507061   | 0.037940243893973456 | 0.8353168089297439    | 0.726341707108193  | 0.7770270270270271 | 0.7457554960701812    | 0.5268910593795443 | 0.5957102913175522 | 0.8008094825093958  | 0.783149561775516  | 0.7918810748999427 | 0.7206364102267594  | 0.5979985403927984  | 0.6379930804561698  | 0.9019893664248397  | 0.6347543807856573  | 0.9247808101480897  | 0.6888280196279318  | 0.8985598215377829  | 0.7220143140903305  |
| 3     | 0.04717027395963669  | 0.03812219600794536  | 0.8460403153765488    | 0.7772980302597773 | 0.8102136839560358 | 0.7627070198310643    | 0.623566106925357  | 0.6791124581829259 | 0.7867718278677183  | 0.8037885213457733 | 0.7951891476120552 | 0.6840651982996002  | 0.6608704063336416  | 0.6672605949996667  | 0.9098888616053247  | 0.680818669580584   | 0.9331399030757054  | 0.7764146090753296  | 0.8977661940152964  | 0.7588442258406161  |

---

### Первая модель (CLS):
- Лучший `val_f1_micro` на 3-й эпохе: **0.7918**
- Лучший `val_f1_macro` на 3-й эпохе: **0.6659**
- Лучший `val_roc_auc_class_*` на 3-й эпохе:
  - Максимальное значение: **0.9263** (Класс 2)
  - Минимальное значение: **0.7701** (Класс 5)

### Вторая модель (CLS_X):
- Лучший `val_f1_micro` на 3-й эпохе: **0.7951**
- Лучший `val_f1_macro` на 3-й эпохе: **0.6673**
- Лучший `val_roc_auc_class_*` на 3-й эпохе:
  - Максимальное значение: **0.9331** (Класс 2)
  - Минимальное значение: **0.7588** (Класс 5)

---

### Итог
1. **По `val_f1_micro`**:
   - CLS_X показала лучшее значение: **0.7951** против **0.7918** у CLS.
   
2. **По `val_f1_macro`**:
   - CLS_X также немного лучше: **0.6673** против **0.6659**.

3. **По `ROC-AUC`**:
   - CLS_X имеет более высокие значения для большинства классов, включая лучший результат **0.9331** для класса 2.
   - Однако минимальное значение у CLS выше: **0.7701** против **0.7588** у CLS_X.

---

### Рекомендация
Модель **CLS_X** в целом показывает лучшие результаты, особенно в метриках `F1` и `ROC-AUC`. Она подходит для задач мультилейбл классификации, где важна высокая предсказательная способность для большинства классов.

### Выводы:
1. `val_f1_micro`: Вторая модель (CLS_X) немного лучше (0.7951 против 0.7918 у CLS).
2. `val_f1_macro`: Вторая модель (CLS_X) также немного лучше (0.6673 против 0.6659 у CLS).

Это указывает на то, что вторая модель (CLS_X) лучше справляется с общей предсказательной способностью, включая как часто встречающиеся, так и редкие классы.

Дополнительный фактор:
- **ROC-AUC**: Вторая модель показывает более высокие значения для большинства классов (максимальное 0.9331), хотя минимальное значение ROC-AUC у первой модели выше (0.7701 против 0.7588 у CLS_X).
- Потери на валидации (`val_loss`) у обеих моделей схожи, но CLS_X немного лучше минимизирует потери.

### Итог:
Вторая модель (CLS_X) справилась лучше, так как достигла лучших значений `F1` (как Micro, так и Macro) и имеет более высокие значения ROC-AUC для большинства классов. Это делает ее предпочтительным выбором для задач мультилейбл классификации.

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
