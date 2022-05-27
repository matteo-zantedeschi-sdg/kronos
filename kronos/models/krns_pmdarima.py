from mlflow.tracking import MlflowClient
import pmdarima as pm
import pandas as pd
import logging
import mlflow
import datetime

logger = logging.getLogger(__name__)


class KRNSPmdarima:
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
        test_data: pd.DataFrame = pd.DataFrame(),
        model: pm.arima.arima.ARIMA = None,
        m: int = 7,
        seasonal: bool = True,
    ):
        # Kronos attributes
        self.key_column = key_column
        self.date_col = date_col
        self.metric_col = metric_col
        self.fcst_col = fcst_col
        self.train_data = train_data.copy()
        self.test_data = test_data.copy()

        # Model attributes
        self.m = m
        self.seasonal = seasonal

        # To load an already configured model
        self.model = model

        self.model_params = {"m": self.m, "seasonal": self.seasonal}

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
            mlflow.pmdarima.log_model(pr_model=self.model, artifact_path=artifact_path)
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
