"""This module contains utils functions"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score
import shap


def validate_model(model: object, x_test: pd.DataFrame, y_test: pd.Series):
    """Function to validate model"""

    y_pred = model.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    print(f'Accuracy Score : {acc * 100 : .2f}%')

    roc = roc_auc_score(y_test, y_pred)
    print(f'AUC ROC : {roc * 100 : .2f} %')

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return acc


def plot_feature_importance(model: object):
    """Plot feature importance"""

    feature_importance = model.feature_importances_
    feature_names = model.feature_names_
    sorted_idx = np.argsort(feature_importance)

    fig = plt.figure(figsize=(8, 4))
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], align='center')
    plt.yticks(range(len(sorted_idx)), np.array(feature_names)[sorted_idx])
    plt.title('Catboost feature importance')
    plt.show()

    return fig


def plot_shap(model: object, x_train: pd.Series):
    """Plot SHAPLEY"""

    fig = plt.figure(figsize=(8, 4))
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_train)
    shap.summary_plot(shap_values, x_train)
    plt.title('Shap feature importance')
    plt.show()

    return fig
