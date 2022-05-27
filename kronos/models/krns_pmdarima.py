from mlflow.tracking import MlflowClient
import pmdarima as pm
import pandas as pd
import logging
import mlflow
import datetime

logger = logging.getLogger(__name__)


class KRNSPmdarima:
    """
    Class to implement pm.arima.arima.ARIMA in kronos.
    """

    def __init__(
        self,
        key_col: str,
        date_col: str,
        metric_col: str,
        fcst_col: str,
        train_data: pd.DataFrame = pd.DataFrame(),
        test_data: pd.DataFrame = pd.DataFrame(),
        model: pm.arima.arima.ARIMA = None,
        m: int = 7,
        seasonal: bool = True,
    ) -> None:
        """
        Initialization method.

        :param str key_col: The name of the column indicating the time series key.
        :param str date_col: The name of the column indicating the time dimension.
        :param str metric_col: The name of the column indicating the dimension to forecast.
        :param str fcst_col: The name of the column indication the forecast.
        :param pd.DataFrame train_data: Pandas DataFrame with the training data.
        :param pd.DataFrame test_data: Pandas DataFrame with the test data.
        :param pm.arima.arima.ARIMA model: An already fitted ARIMA model, to instantiate a kronos Pmdarima from an already fitted model.
        :param int m: The period for seasonal differencing, m refers to the number of periods in each season.
        :param bool seasonal: Whether to fit a seasonal ARIMA.

        :return: No return.

        **Example**

        .. code-block:: python

            model = KRNSPmdarima(
                    train_data=df_train,
                    test_data=df_test,
                    key_col='id',
                    date_col='date',
                    metric_col='y',
                    fcst_col='y_hat',
                    m=7,
                    seasonal=True,
                    model=None,
                )

        """
        # Kronos attributes
        self.key_col = key_col
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

    def preprocess(self) -> None:
        """
        Get the dataframe into the condition to be processed by the model: keep only the *date* and *metric* column.

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
            # TODO: Signature to add before log the model
            mlflow.pmdarima.log_model(
                pmdarima_model=self.model, artifact_path=artifact_path
            )
            logger.info(f"### Model logged: {self.model}")

        except Exception as e:
            logger.error(f"### Log model {self.model} failed: {e}")

    def fit(self) -> None:
        """
        Instantiate the model class and fit the model.

        :return: No return.
        """
        # TODO: Is it possible to add a max/min (saturating maximum and minimum) value during training.
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
    ) -> pd.DataFrame:
        """
        Predict using the fitted model.

        :param int n_days: Number of data points to predict.
        :param datetime.date fcst_first_date: First date of forecast.

        :return: *(pd.DataFrame)* Pandas DataFrame containing the predictions.
        """

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
