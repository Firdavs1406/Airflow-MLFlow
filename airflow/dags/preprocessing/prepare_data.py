"""This module is used to prepare data for model training"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from catboost import Pool


def prepare_data(data: pd.DataFrame, test_size: float, random_state: int = 42) -> pd.DataFrame:
    """Function to prepare data"""

    #drop unnecessary columns
    unnecessary_cols = ['nameOrig', 'nameDest', 'isFlaggedFraud', 'step']
    data.drop(unnecessary_cols, axis=1, inplace=True)

    # encode categorical variable
    label_encoder = LabelEncoder()
    data['type'] = label_encoder.fit_transform(data['type'])

    # split data into features (X) and target (y)
    x = data.drop(['isFraud'], axis = 1)
    y = data['isFraud']

    x_scaled = scale_feautres(x)

    # split data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y, test_size=test_size, random_state=random_state
    )

    print(f'Train size is: {x_train.shape} \nTest size is: {x_test.shape}')

    return x_train, x_test, y_train, y_test


def scale_feautres(data: pd.DataFrame) -> pd.DataFrame:
    """Scale Features with StandartScaler"""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    return pd.DataFrame(scaled_data, columns=data.columns)


def get_pool(x, y):
    """Make Pool"""
    data_pool = Pool(x, y)

    return data_pool
