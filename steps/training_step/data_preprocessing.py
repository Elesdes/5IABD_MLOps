import pandas as pd
from typing_extensions import Annotated
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from zenml import step


@step()
def data_preprocessing(
    df: pd.DataFrame,
) -> tuple[
    Annotated[pd.DataFrame, "x_train"],
    Annotated[pd.DataFrame, "x_test"],
    Annotated[pd.DataFrame, "y_train"],
    Annotated[pd.core.series.Series, "y_test"],
    Annotated[pd.DataFrame, "x"],
]:
    """
    Dataframe preprocessing step.

    :param df: A pandas DataFrame that contains the data retrieved
    :return:
    The training dataset and the testing dataset. The training output and the testing output to predict.
    """
    x = df.iloc[:, :80].copy()
    y = df.iloc[:, 80].copy()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=1
    )
    SC = StandardScaler()
    x_train = pd.DataFrame(SC.fit_transform(x_train))
    x_test = pd.DataFrame(SC.transform(x_test))
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_train = pd.DataFrame(y_train)
    return x_train, x_test, y_train, y_test, x
