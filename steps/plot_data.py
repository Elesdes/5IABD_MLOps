import pandas as pd
import matplotlib.pyplot as plt
from zenml import step


@step()
def plot_data(data: pd.DataFrame) -> None:
    """
    Dataset viewer step
    :param data: A pandas DataFrame that contains the data that will be shown as an electromyogram    :return:
    Nothing
    """
    fig, axes = plt.subplots(2, 4, figsize=(30, 8), sharex=True,
                             sharey=True)  # ensures that all subplots share the same x-axis and y-axis
    for i in range(2):
        for j in range(4):
            axes[i][j].plot(data.iloc[:, i * j])
