from prophet import Prophet
from mlflow.tracking import MlflowClient
import pandas as pd
import logging
import mlflow
import datetime

logger = logging.getLogger(__name__)


class KRNSProphet:
    """
    Class to implement Prophet in kronos.
    """

    def __init__(
        self,
        key_col: str,
        date_col: str,
        metric_col: str,
        fcst_col: str,
        train_data: pd.DataFrame = pd.DataFrame(),
        test_data: pd.DataFrame = pd.DataFrame(),
        interval_width: float = 0.95,
        growth: str = "linear",
        daily_seasonality: bool = False,
        weekly_seasonality: bool = True,
        yearly_seasonality: bool = True,
        seasonality_mode: str = "multiplicative",
        floor: int = None,
        cap: int = None,
        country_holidays: str = "IT",
        model: Prophet = None,
    ) -> None:
        """
        Initialization method.

        :param str key_col: The name of the column indicating the time series key.
        :param str date_col: The name of the column indicating the time dimension.
        :param str metric_col: The name of the column indicating the dimension to forecast.
        :param str fcst_col: The name of the column indication the forecast.
        :param pd.DataFrame train_data: Pandas DataFrame with the training data.
        :param pd.DataFrame test_data: Pandas DataFrame with the test data.
        :param float interval_width: Width of the uncertainty intervals provided for the forecast.
        :param str growth: String ’linear’, ’logistic’, or ’flat’ to specify a linear, logistic or flat trend.
        :param bool daily_seasonality: Fit daily seasonality.
        :param bool weekly_seasonality: Fit weekly seasonality.
        :param bool yearly_seasonality: Fit yearly seasonality.
        :param str seasonality_mode: One among 'additive' (default) or 'multiplicative'.
        :param int floor: The saturating minimum of the time-series to forecast.
        :param int cap: The saturating maximum  of the time-series to forecast.
        :param str country_holidays: Name of the country to add holidays for.
        :param Prophet model: An already fitted Prophet model, to instantiate a kronos Prophet from an already fitted model.

        :return: No return.

        **Example**

        .. code-block:: python

            krns_model = KRNSProphet(
                    train_data=df_train,
                    test_data=df_test,
                    key_col='id',
                    date_col='date',
                    metric_col='y',
                    fcst_col='y_hat',
                    interval_width=0.95,
                    growth='logistic',
                    daily_seasonality=False,
                    weekly_seasonality=True,
                    yearly_seasonality=True,
                    seasonality_mode='multiplicative',
                    floor=0,
                    cap=9999999999999,
                    country_holidays='IT',
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
        self.interval_width = interval_width if not model else model.interval_width
        self.growth = growth if not model else model.growth
        self.daily_seasonality = (
            daily_seasonality if not model else model.daily_seasonality
        )
        self.weekly_seasonality = (
            weekly_seasonality if not model else model.weekly_seasonality
        )
        self.yearly_seasonality = (
            yearly_seasonality if not model else model.yearly_seasonality
        )
        self.seasonality_mode = (
            seasonality_mode if not model else model.seasonality_mode
        )
        self.country_holidays = (
            country_holidays if not model else model.country_holidays
        )

        # Floor/Cap
        if floor:
            self.floor = floor
        else:
            self.floor = (
                train_data[metric_col].min() / 10 if train_data.shape[0] > 0 else 0
            )

        if cap:
            self.cap = cap
        else:
            self.cap = (
                train_data[metric_col].max() * 10
                if train_data.shape[0] > 0
                else 1000000000000000000
            )

        # To load an already configured model
        self.model = model

        self.model_params = {
            "interval_width": self.interval_width,
            "growth": self.growth,
            "daily_seasonality": self.daily_seasonality,
            "weekly_seasonality": self.weekly_seasonality,
            "yearly_seasonality": self.yearly_seasonality,
            "seasonality_mode": self.seasonality_mode,
            "floor": self.floor,
            "cap": self.cap,
            "country_holidays": self.country_holidays,
        }

    def preprocess(self) -> None:
        """
        Get the dataframe into the condition to be processed by the model: renaming columns with date as 'ds' and with metric as 'y'.

        :return: No return.
        """

        try:
            self.train_data.rename(
                columns={self.date_col: "ds", self.metric_col: "y"}, inplace=True
            )
        except Exception as e:
            logger.warning(
                f"### Preprocess train data failed: {e} - {self.train_data.head(1)}"
            )

        try:
            self.test_data.rename(
                columns={self.date_col: "ds", self.metric_col: "y"}, inplace=True
            )
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
            mlflow.prophet.log_model(pr_model=self.model, artifact_path=artifact_path)
            logger.info(f"### Model logged: {self.model}")

        except Exception as e:
            logger.error(f"### Log model {self.model} failed: {e}")

    def fit(self) -> None:
        """
        Fit the model:
            1. Instantiate the model class.
            2. Add floor/cap to train data.
            3. Add country holidays.
            4. Fit the model.
            5. Remove floor/cap columns from train data.

        :return: No return.
        """

        try:
            # Define the model
            self.model = Prophet(
                interval_width=self.interval_width,
                growth=self.growth,
                daily_seasonality=self.daily_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                yearly_seasonality=self.yearly_seasonality,
                seasonality_mode=self.seasonality_mode,
            )

            # Add floor and cap
            self.train_data["floor"] = self.floor
            self.train_data["cap"] = self.cap

            # Add country holidays
            self.model.add_country_holidays(country_name=self.country_holidays)

            # Fit the model
            self.model.fit(self.train_data)

            # Remove floor and cap
            self.train_data.drop(["floor", "cap"], axis=1, inplace=True)

        except Exception as e:
            logger.error(
                f"### Fit with model {self.model} failed: {e} - on data {self.train_data.head(1)}"
            )

    def predict(
        self, n_days: int, fcst_first_date: datetime.date = datetime.date.today()
    ) -> pd.DataFrame:
        """

        :param int n_days: Number of data points to predict.
        :param datetime.date fcst_first_date: First date of forecast.

        :return: *(pd.DataFrame)* Pandas DataFrame containing the predictions.
        """

        try:
            # Compute difference from last date in the model and first date of forecast
            model_last_date = self.model.history_dates[0].date()
            difference = (fcst_first_date - model_last_date).days

            # configure predictions
            pred_config = self.model.make_future_dataframe(
                periods=difference + n_days - 1, freq="d", include_history=False
            )

            # Add floor and cap
            pred_config["floor"] = self.floor
            pred_config["cap"] = self.cap

            # Make predictions
            pred = self.model.predict(pred_config)

            # Convert datetime to date
            pred["ds"] = pred["ds"].dt.date

            # Keep only relevant period
            pred = pred[pred["ds"] >= fcst_first_date]

            # Rename columns
            pred.rename(
                columns={"ds": self.date_col, "yhat": self.fcst_col}, inplace=True
            )

            return pred

        except Exception as e:
            logger.error(f"### Predict with model {self.model} failed: {e}")
