from typing_extensions import Annotated
import pandas as pd
from zenml import step


@step()
def load_data() -> Annotated[pd.DataFrame, "clean_features"]:
    """
    Dataset reader steps.

    This steps will read the data from csv files locally.
    :return:
    The dataset pre-cleaned
    """
    df = pd.read_csv("./data/emg_all_features_labeled.csv")
    return df
