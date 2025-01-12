# Airflow + MLFlow Project

Этот проект демонстрирует, как интегрировать Apache Airflow и MLFlow для управления и отслеживания машинного обучения.

## Установка

Следуйте этим шагам для установки необходимых компонентов:

### Шаг 1: Клонирование репозитория

```bash
git clone https://github.com/yourusername/Airflow-MLFlow.git
cd Airflow-MLFlow
```

### Шаг 2: Установка виртуального окружения

```bash
python3 -m venv venv
source venv/bin/activate
```

### Шаг 3: Установка зависимостей

```bash
pip install -r requirements.txt
```

### Шаг 4: Настройка Airflow

```bash
export AIRFLOW_HOME=$(pwd)/airflow
airflow db init
```

### Шаг 5: Создание учетной записи
```bash
airflow users create \   
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password 1234
```

### Шаг 6: Настройка MLFlow

Создайте директорию для хранения данных MLFlow:

```bash
mkdir mlflow
export MLFLOW_REGISTRY_URI=mlflow
```
Полезная инфа: https://www.mlflow.org/docs/latest/tracking.html#tracking-ui


## Запуск

### Шаг 1: Запуск Airflow

Запустите веб-сервер и планировщик Airflow:

```bash
airflow webserver --port 8080
airflow scheduler
```

### Шаг 2: Запуск MLFlow

Запустите сервер MLFlow:

```bash
mlflow server --host localhost --port 5000 --backend-store-uri sqlite:///${MLFLOW_REGISTRY_URI}/mlflow.db --default-artifact-root ${MLFLOW_REGISTRY_URI}
```

### Шаг 3: Открытие интерфейсов

- Airflow: [http://localhost:8080](http://localhost:8080)
- MLFlow: [http://localhost:5000](http://localhost:5000)


## Описание DAG

### 1. **prepare_data_task**

Этот шаг отвечает за подготовку данных для обучения модели. Он включает в себя:

- Считывание данных из CSV файла.
- Разделение данных на обучающую и тестовую выборки.
- Сериализацию данных для передачи между задачами с помощью XCom.

### 2. **train_model_task**

Этот шаг обучает модель с использованием **CatBoostClassifier** и **Optuna**. Включает следующие этапы:

- Определение функции `objective` для оптимизации гиперпараметров модели.
- Использование **Optuna** для выбора наилучших гиперпараметров (количество итераций, глубина дерева и скорость обучения).
- Обучение модели с выбранными гиперпараметрами.
- Логирование точности модели, гиперпараметров и важности признаков с помощью **MLflow**.

## Настройки и параметры

В файле `train_model.py` доступны следующие параметры:

- `DATA_PATH`: Путь к данным (CSV файл).
- `MFLOW_TRACKING_URI`: Адрес для отслеживания экспериментов в MLflow.
- `EXPERIMENT_NAME`: Имя эксперимента в MLflow.
- `RANDOM_STATE`: Значение для контроля случайности.
- `TEST_SIZE`: Размер тестовой выборки.

## Логирование

Результаты обучения модели, параметры и важность признаков логируются в **MLflow**.

## Примечания

- Параметры для обучения модели (например, гиперпараметры CatBoost) могут быть изменены в коде.
- Вы можете добавлять новые задачи или изменять текущие шаги DAG для улучшения или расширения функциональности.
