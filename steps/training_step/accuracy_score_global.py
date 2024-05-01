import pandas as pd
from zenml import step
from sklearn.metrics import accuracy_score
import numpy


@step()
def accuracy_score_global(y_xgb: numpy.ndarray, y_randomforest: numpy.ndarray, y_test: pd.DataFrame) -> tuple[float, float, str]:
    """
    Accuracy score validation step
    :param y_xgb: The predicted output of the XGB model
    :param y_randomforest: The predicted output of the Random Forest model
    :param y_test: The testing dataframe output
    :return:
    Double accuracies of XGB and RandomForest and the string for discord
    """
    accuracy_xgb = accuracy_score(y_test, [r for r in y_xgb]) * 100.0
    accuracy_randomforest = accuracy_score(y_test, [q for q in y_randomforest])* 100.0
    print("Accuracy for XGB: %.2f%%" % (accuracy_xgb))
    print("Accuracy for RF: %.2f%%" % (accuracy_randomforest))
    return accuracy_xgb, accuracy_randomforest, "Ended correctly with Accuracy for XGB: %.2f%% and Accuracy for RF: %.2f%%" % (accuracy_xgb * 100.0, accuracy_randomforest * 100.0)
