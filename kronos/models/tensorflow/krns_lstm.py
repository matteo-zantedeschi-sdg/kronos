from mlflow.tracking import MlflowClient
import os
import copy
import tensorflow as tf
import pandas as pd
import numpy as np
import logging
import mlflow
import datetime

logger = logging.getLogger(__name__)


class KRNSLSTM:
    """
    Class to implement LSTM in kronos.
    """

    def __init__(
        self,
        modeler,  # TODO: How to explicit its data type without incur in [...] most likely due to a circular import
        model: tf.keras.Sequential = None,
        n_units: int = 128,
        activation: str = "relu",
        epochs: int = 25,
        n_inputs: int = 30,
    ) -> None:
        """
        Initialization method.

        :param Modeler modeler: The Modeler instance used to interact with data.
        :param tf.keras.Sequential: An already fitted Sequential model containing a LSTM layer, to instantiate a kronos LSTM from an already fitted model.
        :param int n_units: Number of units in the LSTM layer of the model.
        :param str activation: Activation function in the LSTM layer of the model.
        :param int epochs: Number of epochs to train the model for.
        :param int n_inputs: Number of lag considered for the training of one step ahead.

        :return: No return.

        **Example**

        .. code-block:: python

            model = KRNSLSTM(
                    modeler=modeler,
                    n_units=128,
                    activation='relu',
                    epochs=25,
                    n_inputs=30,
                    model=None,
                )

        """
        # Kronos attributes
        self.modeler = copy.deepcopy(modeler)

        # Model attributes
        self.n_inputs = n_inputs
        self.activation = activation
        self.epochs = epochs
        self.n_units = n_units

        # To load an already configured model
        self.model = model

        self.model_params = {
            "n_inputs": self.n_inputs,
            "activation": self.activation,
            "epochs": self.epochs,
            "n_units": self.n_units,
        }

    def preprocess(self) -> None:
        """
        Get the dataframe into the condition to be processed by the model: transform the data into a numpy array.

        :return: No return.
        """

        try:
            self.modeler.train_data = np.array(
                self.modeler.train_data[self.modeler.metric_col]
            )
        except Exception as e:
            logger.warning(f"### Preprocess train data failed: {e}")

        try:
            self.modeler.test_data = np.array(
                self.modeler.test_data[self.modeler.metric_col]
            )
        except Exception as e:
            logger.warning(f"### Preprocess test data failed: {e}")

    def log_params(self, client: MlflowClient, run_id: str) -> None:
        """
        Log the model params to mlflow.

        :param MlflowClient client: The mlflow client used to log parameters.
        :param str run_id: The run id under which log parameters.

        :return: No return.
        """
        try:
            for key, val in self.model_params.items():
                client.log_param(run_id, key, val)
        except Exception as e:
            logger.error(f"### Log params {self.model_params} failed: {e}")

    def log_model(self, artifact_path: str) -> None:
        """
        Log the model artifact to mlflow.

        :param str artifact_path: Run-relative artifact path.

        :return: No return.
        """
        try:
            # Save the model
            saved_model_path = os.path.join(os.getcwd(), self.modeler.key_code)
            tf.saved_model.save(self.model, saved_model_path)

            # Define graph tags and signature key
            tag = [tf.saved_model.SERVING]
            key = tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY

            # TODO: Signature to add before log the model
            mlflow.tensorflow.log_model(
                tf_saved_model_dir=saved_model_path,
                artifact_path=artifact_path,
                tf_meta_graph_tags=tag,
                tf_signature_def_key=key,
            )

            logger.info(f"### Model logged: {self.model}")

        except Exception as e:
            logger.error(f"### Log model {self.model} failed: {e}")

    def fit(self) -> None:
        """
        Instantiate the Sequential model class, compile and fit the model with TimeseriesGenerator.

        :return: No return.
        """
        try:
            # Define the model
            self.model = tf.keras.Sequential(
                [
                    tf.keras.layers.Input(shape=(self.n_inputs, 1)),
                    tf.keras.layers.LSTM(
                        units=self.n_units, activation=self.activation
                    ),
                    tf.keras.layers.Dense(units=1),
                ]
            )

            # Compile the model
            self.model.compile(optimizer="adam", loss="mse")

            # Define generator
            ts_generator = tf.keras.preprocessing.sequence.TimeseriesGenerator(
                data=self.modeler.train_data,
                targets=self.modeler.train_data,
                length=self.n_inputs,
                batch_size=1,
            )

            # Fit the model
            self.model.fit(
                ts_generator, steps_per_epoch=len(ts_generator), epochs=self.epochs
            )

        except Exception as e:
            logger.error(
                f"### Fit with model {self.model} failed: {e} - on data {self.modeler.train_data.head(1)}"
            )

    def predict(
        self, n_days: int, fcst_first_date: datetime.date = datetime.date.today()
    ) -> pd.DataFrame:
        """
        Predict using the fitted model.
        Within the body of the function, the predict method is only called on newly trained models that are still in memory.
        For serialized models from the mlflow model register another prediction method is used.

        :param int n_days: Number of data points to predict.
        :param datetime.date fcst_first_date: First date of forecast.

        :return: *(pd.DataFrame)* Pandas DataFrame containing the predictions.
        """

        try:
            # Keep only historic data
            historic_data = self.modeler.data[
                self.modeler.data[self.modeler.date_col] < fcst_first_date
            ]

            # Preprocess historic data
            historic_data = np.array(historic_data[self.modeler.metric_col])

            # Autoregressive prediction
            predictions = []
            batch = historic_data.astype("float32")[-self.n_inputs :].reshape(
                (1, self.n_inputs, 1)
            )
            for i in range(n_days):
                # Get the prediction value for the first batch: we need to differentiate when we directly use the model after training or when we load it from mlflow.
                if type(self.model) == tf.keras.Sequential:
                    # Model directly used after training
                    pred_val = self.model.predict(batch)[0]
                else:
                    # Model loaded from mlflow model registry
                    pred_val = self.model(input_1=batch)["dense"].numpy()[0]

                # Append the prediction into the array
                predictions.append(pred_val[0])

                # Use the prediction to update the batch and remove the first value
                batch = np.append(batch[:, 1:, :], [[pred_val]], axis=1)

            # Make predictions dataframe
            pred = pd.DataFrame(
                data={
                    self.modeler.date_col: [
                        fcst_first_date + datetime.timedelta(days=x)
                        for x in range(n_days)
                    ],
                    self.modeler.fcst_col: predictions,
                }
            )

            return pred

        except Exception as e:
            logger.error(f"### Predict with model {self.model} failed: {e}")
