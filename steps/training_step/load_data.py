from typing_extensions import Annotated
import pandas as pd
from zenml import step


@step()
def load_data() -> tuple[Annotated[pd.DataFrame, "clean_features"], Annotated[pd.DataFrame, "raw_features"]]:
    """
    Dataset reader steps.

    This steps will read the data from csv files locally.
    :return:
    The dataset pre-cleaned and the raw dataset
    """
    df = pd.read_csv('./data/emg_all_features_labeled.csv')
    raw = pd.read_csv('./data/index_finger_motion_raw.csv')
    return df, raw