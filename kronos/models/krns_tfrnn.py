from mlflow.tracking import MlflowClient
import logging
import pandas as pd
import datetime
import mlflow
import tensorflow
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

logger = logging.getLogger(__name__)


class KRNSTfrnn:
    """
    # TODO: Doc
    """

    def __init__(
        self,
        key_column: str,
        date_col: str,
        metric_col: str,
        fcst_col: str,
        train_data: pd.DataFrame = pd.DataFrame(),
        test_data: pd.DataFrame = pd.DataFrame()
        # model: tf.arima.arima.ARIMA = None,
        # ,dropout: float = 0.1,
        # recurrent_dropout: float = 0.2,
        # return_sequences: bool = True,
        # n_epochs: int = 500,
        # batch_size: int = 40
    ):
        # Kronos attributes
        self.key_column = key_column
        self.date_col = date_col
        self.metric_col = metric_col
        self.fcst_col = fcst_col
        self.train_data = train_data.copy()
        self.test_data = test_data.copy()

        # Model attributes
        # self.dropout = dropout
        # self.recurrent_dropout = recurrent_dropout
        # self.return_sequences = return_sequences
        # self.n_epochs = n_epochs
        # self.batch_size = batch_size

        # To load an already configured model
        # self.model = model

        # self.model_params = {"dropout": self.dropout,
        #                      "recurrent_dropout": self.recurrent_dropout,
        #                      "return_sequences": self.return_sequences,
        #                      "n_epochs": self.n_epochs,
        #                      "batch_size": self.batch_size}

    def preprocess(self):
        """
        Get the dataframe into the condition to be processed by the model.
        :return: No return.
        """

        try:
            self.train_data.drop(
                self.train_data.columns.difference([self.date_col, self.metric_col]),
                axis=1,
                inplace=True,
            )
            self.train_data.set_index(self.date_col, inplace=True)


        except Exception as e:
            logger.warning(
                f"### Preprocess train data failed: {e} - {self.train_data.head(1)}"
            )

        try:
            self.test_data.drop(
                self.test_data.columns.difference([self.date_col, self.metric_col]),
                axis=1,
                inplace=True,
            )
            self.test_data.set_index(self.date_col, inplace=True)
        except Exception as e:
            logger.warning(
                f"### Preprocess test data failed: {e} - {self.test_data.head(1)}"
            )

    def log_params(self, client: MlflowClient, run_id: str):
        """
        # TODO: Doc
        """
        try:
            for key, val in self.model_params.items():
                client.log_param(run_id, key, val)
        except Exception as e:
            logger.error(f"### Log params {self.model_params} failed: {e}")

    def log_model(self, artifact_path: str):
        """
        TODO: Doc
        """
        try:
            # Get the model signature and log the model
            # signature = infer_signature(train_data, krns_pmdarima.predict(n_days=n_test))
            # TODO: Signature da aggiungere in futuro, e capire quale
            mlflow.tensorflow.log_model(pr_model=self.model, artifact_path=artifact_path)
            logger.info(f"### Model logged: {self.model}")

        except Exception as e:
            logger.error(f"### Log model {self.model} failed: {e}")

    def fit(self):
        # TODO: Capire se esiste un modo per dargli un min/max value
        try:
            # Define the model
            self.model = pm.auto_arima(
                self.train_data, seasonal=self.seasonal, m=self.m
            )

        except Exception as e:
            logger.error(
                f"### Fit with model {self.model} failed: {e} - on data {self.train_data.head(1)}"
            )

    def predict(
        self, n_days: int, fcst_first_date: datetime.date = datetime.date.today()
    ):

        try:
            # make predictions
            pred = pd.DataFrame(
                data={
                    self.date_col: [
                        fcst_first_date + datetime.timedelta(days=x)
                        for x in range(n_days)
                    ],
                    self.fcst_col: self.model.predict(n_periods=n_days),
                }
            )

            return pred

        except Exception as e:
            logger.error(f"### Predict with model {self.model} failed: {e}")
