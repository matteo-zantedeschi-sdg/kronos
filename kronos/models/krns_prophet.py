from prophet import Prophet
from mlflow.tracking import MlflowClient
import copy
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
        modeler,  # TODO: How to explicit its data type without incur in [...] most likely due to a circular import
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

        :param Modeler modeler: The Modeler instance used to interact with data.
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
                    modeler=modeler,
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
        self.modeler = copy.deepcopy(modeler)

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
                self.modeler.train_data[self.modeler.metric_col].min() / 10
                if self.modeler.train_data.shape[0] > 0
                else 0
            )

        if cap:
            self.cap = cap
        else:
            self.cap = (
                self.modeler.train_data[self.modeler.metric_col].max() * 10
                if self.modeler.train_data.shape[0] > 0
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
            self.modeler.data.rename(
                columns={self.modeler.date_col: "ds", self.modeler.metric_col: "y"},
                inplace=True,
            )
        except Exception as e:
            logger.warning(
                f"### Preprocess data failed: {e} - {self.modeler.data.head(1)}"
            )

        try:
            self.modeler.train_data.rename(
                columns={self.modeler.date_col: "ds", self.modeler.metric_col: "y"},
                inplace=True,
            )
        except Exception as e:
            logger.warning(
                f"### Preprocess train data failed: {e} - {self.modeler.train_data.head(1)}"
            )

        try:
            self.modeler.test_data.rename(
                columns={self.modeler.date_col: "ds", self.modeler.metric_col: "y"},
                inplace=True,
            )
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
            self.modeler.train_data["floor"] = self.floor
            self.modeler.train_data["cap"] = self.cap

            # Add country holidays
            self.model.add_country_holidays(country_name=self.country_holidays)

            # Fit the model
            self.model.fit(self.modeler.train_data)

            # Remove floor and cap
            self.modeler.train_data.drop(["floor", "cap"], axis=1, inplace=True)

        except Exception as e:
            logger.error(
                f"### Fit with model {self.model} failed: {e} - on data {self.modeler.train_data.head(1)}"
            )

    @staticmethod
    def stan_init(m: Prophet) -> dict:
        """
        Retrieve parameters from a trained model in the format used to initialize a new Stan model.

        :param Prophet m: A trained model of the Prophet class.

        :return: (*dict*) A Dictionary containing retrieved parameters of m.
        """
        try:
            res = {}
            for pname in ["k", "m", "sigma_obs"]:
                res[pname] = m.params[pname][0][0]
            for pname in ["delta", "beta"]:
                res[pname] = m.params[pname][0]
            return res

        except Exception as e:
            logger.error(f"### stan_init with model {m} failed: {e}")

    def update_model(self, df_update: pd.DataFrame) -> Prophet:
        """
        Method to update the already fitted model with new data.
        Update in this case is intended as a new fit with warm-start.

        :param DataFrame df_update: The dataframe containing the data to update the model with.

        :return: (*Prophet*) The updated Prophet model.
        """

        try:
            # Define the model
            model = Prophet(
                interval_width=self.model.interval_width,
                growth=self.model.growth,
                daily_seasonality=self.model.daily_seasonality,
                weekly_seasonality=self.model.weekly_seasonality,
                yearly_seasonality=self.model.yearly_seasonality,
                seasonality_mode=self.model.seasonality_mode,
            )

            # Add floor and cap
            df_update["floor"] = self.floor
            df_update["cap"] = self.cap

            # Add country holidays
            model.add_country_holidays(country_name=self.model.country_holidays)

            # Fit the model with warm start
            model.fit(df_update, init=self.stan_init(self.model))

            # Remove floor and cap
            df_update.drop(["floor", "cap"], axis=1, inplace=True)

            return model

        except Exception as e:
            logger.error(
                f"### Update model {self.model} with data {df_update.head(1)} failed: {e}"
            )

    def predict(
        self,
        n_days: int,
        fcst_first_date: datetime.date = datetime.date.today(),
        future_only: bool = True,
        test: bool = False,
        return_conf_int: bool = True

    ) -> pd.DataFrame:
        """
        Predict using the fitted model.

        Four situations can occur:
            1. fcst_first_date <= last_training_day and difference < n_days (still something to forecast) - Note: actual data used as forecast.
            2. fcst_first_date << last training day and difference >= n_days (nothing to forecast) - Note: actual data used as forecast.
            3. fcst_first_date > last training day and some available intermediate data - Note: model update.
            4. fcst_first_date > last training day and no intermediate data available.

        Finally, depending on the parameter *Modeler.future_only*, it is decided whether to keep only the observations from fcst_first_date onwards or also those in between.

        :param int n_days: Number of data points to predict.
        :param datetime.date fcst_first_date: First date of forecast.
        :param bool future_only: Whether to return predicted missing values between the last observed date and the forecast first date (*False*) or only future values (*True*), i.e. those from the forecast first date onwards.
        :param bool test: Wheter to collect x-reg from test data, or from pred_data


        :return: *(pd.DataFrame)* Pandas DataFrame containing the predictions.
        """

        try:
            # Preprocess data (if needed)
            if (
                "ds" not in self.modeler.data.columns
                or "y" not in self.modeler.data.columns
            ):
                self.preprocess()

            # Retrieve model last training day
            last_training_day = self.model.history_dates.max().date()

            # Update model with last data (if any) and update last_training_day value
            n_update_rows = len(
                self.modeler.data[
                    (last_training_day < self.modeler.data["ds"])
                    & (self.modeler.data["ds"] < fcst_first_date)
                ]
            )
            if n_update_rows > 0:
                update_data = self.modeler.data[
                    self.modeler.data["ds"] < fcst_first_date
                ]
                self.model = self.update_model(df_update=update_data)
                last_training_day = self.model.history_dates.max().date()

            # Compute the difference between last_training_day and fcst_first_date
            difference = (fcst_first_date - last_training_day).days

            # Set include_history based on whether fcst_first_date is older or newer than last_training_day
            include_history = False if difference > 0 else True

            # Compute actual forecast horizon
            fcst_horizon = max(difference + n_days - 1, 0)

            # Configure predictions
            pred_config = self.model.make_future_dataframe(
                periods=fcst_horizon, freq="d", include_history=include_history
            )

            # Add floor and cap
            pred_config["floor"] = self.floor
            pred_config["cap"] = self.cap

            # Make predictions
            pred = self.model.predict(pred_config)

            # Convert datetime to date
            pred["ds"] = pred["ds"].dt.date

            # Keep relevant data
            if future_only:
                pred = pred[pred["ds"] >= fcst_first_date]
            if difference < 0:
                pred = pred[
                    pred["ds"] < fcst_first_date + datetime.timedelta(days=n_days)
                ]

            # Rename columns
            pred.rename(
                columns={"ds": self.modeler.date_col, "yhat": self.modeler.fcst_col},
                inplace=True,
            )

            return pred

        except Exception as e:
            logger.error(f"### Predict with model {self.model} failed: {e}")
