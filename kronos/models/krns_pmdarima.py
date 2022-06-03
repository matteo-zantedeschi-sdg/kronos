from mlflow.tracking import MlflowClient
import copy
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
        modeler,  # TODO: How to explicit its data type without incur in [...] most likely due to a circular import
        model: pm.arima.arima.ARIMA = None,
        m: int = 7,
        seasonal: bool = True,
    ) -> None:
        """
        Initialization method.

        :param Modeler modeler: The Modeler instance used to interact with data.
        :param pm.arima.arima.ARIMA model: An already fitted ARIMA model, to instantiate a kronos Pmdarima from an already fitted model.
        :param int m: The period for seasonal differencing, m refers to the number of periods in each season.
        :param bool seasonal: Whether to fit a seasonal ARIMA.

        :return: No return.

        **Example**

        .. code-block:: python

            model = KRNSPmdarima(
                    modeler=modeler,
                    m=7,
                    seasonal=True,
                    model=None,
                )

        """
        # Kronos attributes
        self.modeler = copy.deepcopy(modeler)

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
            self.modeler.train_data.drop(
                self.modeler.train_data.columns.difference(
                    [self.modeler.date_col, self.modeler.metric_col]
                ),
                axis=1,
                inplace=True,
            )
            self.modeler.train_data.set_index(self.modeler.date_col, inplace=True)
        except Exception as e:
            logger.warning(
                f"### Preprocess train data failed: {e} - {self.modeler.train_data.head(1)}"
            )

        try:
            self.modeler.test_data.drop(
                self.modeler.test_data.columns.difference(
                    [self.modeler.date_col, self.modeler.metric_col]
                ),
                axis=1,
                inplace=True,
            )
            self.modeler.test_data.set_index(self.modeler.date_col, inplace=True)
        except Exception as e:
            logger.warning(
                f"### Preprocess test data failed: {e} - {self.modeler.test_data.head(1)}"
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
                self.modeler.train_data, seasonal=self.seasonal, m=self.m
            )

            # Add last training day attribute
            self.model.last_training_day = self.modeler.train_data.index.max()

        except Exception as e:
            logger.error(
                f"### Fit with model {self.model} failed: {e} - on data {self.modeler.train_data.head(1)}"
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
            # Preprocess data (if needed)
            if type(self.modeler.data.index[0]) != datetime.date:
                self.preprocess()

            # Retrieve model last training day
            last_training_day = copy.deepcopy(self.model.last_training_day)

            # Update model with last data (if any)
            update_data = self.modeler.data[
                last_training_day < self.modeler.data.index < fcst_first_date
            ]
            if len(update_data) > 0:
                self.model.update(update_data)
                last_training_day = update_data.index.max()

            # Compute the difference between last_training_day and fcst_first_date
            difference = (fcst_first_date - last_training_day).days

            # Compute actual forecast horizon
            fcst_horizon = difference + n_days - 1

            # make predictions
            pred = pd.DataFrame(
                data={
                    self.modeler.date_col: [
                        fcst_first_date + datetime.timedelta(days=x)
                        for x in range(fcst_horizon)
                    ],
                    self.modeler.fcst_col: self.model.predict(n_periods=fcst_horizon),
                }
            )

            return pred

        except Exception as e:
            logger.error(f"### Predict with model {self.model} failed: {e}")
