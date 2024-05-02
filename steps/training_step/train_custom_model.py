import tensorflow as tf
import pandas as pd
from zenml import step


@step()
def train_custom_model(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x: pd.DataFrame,
    batch_size: int,
    epochs: int,
) -> None:
    """
    The training step of the custom tensorflow model
    :param x_train: The dataframe input for the prediction
    :param y_train: The dataframe input to achieve
    :param batch_size: An integer to know the batch size. Multiples of 8 are recommended.
    :param epochs: An integer to know how many time the model will read the input data.
    :return:
    The Random Forest model and the predicted data from the test dataframe
    """
    inputs = tf.keras.Input(shape=(x.shape[1],))
    x = tf.keras.layers.Dense(80, activation="relu")(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(80, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(8, activation="softmax")(x)

    model_cnn = tf.keras.Model(inputs, outputs)

    model_cnn.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    history = model_cnn.fit(
        x_train,
        y_train,
        validation_split=0.2,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[tf.keras.callbacks.ReduceLROnPlateau()],
    )
    # History ne peut pas être passé en l'état puisque ce n'est pas un objet supporté par le framework
    # Il faut penser à le convertir d'une manière ou d'une autre.
    # Il en va de même pour les callbacks et le modèle

    # return model_cnn
