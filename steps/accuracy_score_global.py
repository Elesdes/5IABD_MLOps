import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from zenml import step
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
import numpy


@step()
def accuracy_score_global(y_xgb: numpy.ndarray, y_randomforest: numpy.ndarray, y_test: pd.DataFrame) -> tuple[float, float]:
    """
    Accuracy score validation step
    :param y_xgb: The predicted output of the XGB model
    :param y_randomforest: The predicted output of the Random Forest model
    :param y_test: The testing dataframe output
    :return:
    Nothing
    """
    accuracy_xgb = accuracy_score(y_test, [r for r in y_xgb])
    accuracy_randomforest = accuracy_score(y_test, [q for q in y_randomforest])
    print("Accuracy for XGB: %.2f%%" % (accuracy_xgb * 100.0))
    print("Accuracy for RF: %.2f%%" % (accuracy_randomforest * 100.0))
    return accuracy_xgb * 100.0, accuracy_randomforest * 100.0
    """
    plt.plot(range(epochs), history.history['loss'], label="Training Loss")
    plt.plot(range(epochs), history.history['val_loss'], label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss Over Time")

    plt.show()

    y_true = np.array(y_test)
    y_pred = np.array(model_RF.predict(x_test))
    y_pred_xgb = np.array(model_XGB.predict(x_test))

    cm = confusion_matrix(y_true, y_pred)
    cm_xgb = confusion_matrix(y_true, y_pred_xgb)
    plt.figure(figsize=(15, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='gist_heat', cbar=True)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix for RF")
    plt.show()
    plt.figure(figsize=(15, 8))
    sns.heatmap(cm_xgb, annot=True, fmt='g', cmap='gist_heat', cbar=True)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix for XGB")
    plt.show()
    """
