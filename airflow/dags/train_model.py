"""
DAG для обучения модели с помощью MLflow и Optuna с CatBoostClassifier.
Этот скрипт обрабатывает подготовку данных, обучение модели и логирования результатов.

Зависимости:
- pandas
- mlflow
- optuna
- catboost
- airflow
"""
import warnings
from datetime import datetime
import mlflow
import pandas as pd
import optuna
from catboost import CatBoostClassifier

from airflow import DAG
from airflow.operators.python import PythonOperator

from preprocessing.prepare_data import prepare_data
from utils.utils import validate_model, plot_feature_importance

warnings.filterwarnings('ignore')

# Параметры DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
}

dag = DAG(
    'train_model',
    default_args = default_args,
    description = 'DAG для автоматического переобучения модели с использованием MLflow',
    schedule_interval = '*/5 * * * *',
    start_date = datetime(2025, 1, 1),
    catchup = False,
)

# Пути
DATA_PATH = "/Users/firdavs/Desktop/Airfloq/airflow/dags/data/data.csv"
MFLOW_TRACKING_URI = "http://127.0.0.1:8099"
EXPERIMENT_NAME = "optuna_catboost_experiment"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def prepare_data_task(**kwargs):
    """Task for data preparation."""
    data = pd.read_csv(DATA_PATH)
    x_train, x_test, y_train, y_test = prepare_data(data, TEST_SIZE, RANDOM_STATE)

    # Convert pandas objects to lists/dictionaries for serialization
    kwargs['ti'].xcom_push(key='X_train', value=x_train.to_dict('list'))
    kwargs['ti'].xcom_push(key='X_test', value=x_test.to_dict('list'))
    kwargs['ti'].xcom_push(key='y_train', value=y_train.tolist())
    kwargs['ti'].xcom_push(key='y_test', value=y_test.tolist())


def train_model_task(**kwargs):
    """Task for model training."""
    # Retrieve data from previous task
    ti = kwargs['ti']
    x_train_dict = ti.xcom_pull(task_ids='prepare_data_task', key='x_train')
    x_test_dict = ti.xcom_pull(task_ids='prepare_data_task', key='x_test')
    y_train_list = ti.xcom_pull(task_ids='prepare_data_task', key='y_train')
    y_test_list = ti.xcom_pull(task_ids='prepare_data_task', key='y_test')

    # Convert back to pandas objects
    x_train = pd.DataFrame(x_train_dict)
    x_test = pd.DataFrame(x_test_dict)
    y_train = pd.Series(y_train_list)
    y_test = pd.Series(y_test_list)

    mlflow.set_tracking_uri(MFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 100, 1000, step=100),
            'depth': trial.suggest_int('depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'verbose': False
        }
        model = CatBoostClassifier(**params)
        model.fit(x_train, y_train)
        accuracy = validate_model(model, x_test, y_test)
        return accuracy

    with mlflow.start_run():
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=30)
        best_params = study.best_params
        mlflow.log_params(best_params)

        best_model = CatBoostClassifier(**best_params, verbose=False)
        best_model.fit(x_train, y_train)
        accuracy = validate_model(best_model, x_test, y_test)
        mlflow.log_metric("accuracy", accuracy)

        feature_importance_fig = plot_feature_importance(best_model)
        mlflow.log_figure(feature_importance_fig, "feature_importance.png")

prepare_data_op = PythonOperator(
    task_id='prepare_data_task',
    python_callable=prepare_data_task,
    provide_context=True,
    dag=dag,
)

train_model_op = PythonOperator(
    task_id='train_model_task',
    python_callable=train_model_task,
    provide_context=True,
    dag=dag,
)

prepare_data_op >> train_model_op
