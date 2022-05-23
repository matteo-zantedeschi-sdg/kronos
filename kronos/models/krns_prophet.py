from prophet import Prophet
from mlflow.tracking import MlflowClient
import pandas as pd
import logging
import mlflow

logger = logging.getLogger(__name__)


class KRNSProphet:
    """
    # TODO: Doc
    """

    def __init__(
        self,
        key_column: str,
        date_col: str,
        metric_col: str,
        train_data: pd.DataFrame = pd.DataFrame(),
        test_data: pd.DataFrame = pd.DataFrame(),
        interval_width: float = 0.95,
        growth: str = "linear",
        daily_seasonality: bool = False,
        weekly_seasonality: bool = True,
        yearly_seasonality: bool = True,
        seasonality_mode: str = "multiplicative",
        floor: int = 0,
        cap: int = None,
        country_holidays: str = "IT",
        model: Prophet = None,
    ):
        self.key_column = key_column
        self.date_col = date_col
        self.metric_col = metric_col
        self.train_data = train_data
        self.test_data = test_data

        # Model params
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
        self.floor = floor
        self.cap = cap if cap else train_data[metric_col].max() * 10
        self.country_holidays = (
            country_holidays if not model else model.country_holidays
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

    def preprocess(self):
        """
        Get the dataframe into the condition to be processed by the model.
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
            # signature = infer_signature(train_data, krns_prophet.predict(n_days=n_test))
            # TODO: Signature da aggiungere in futuro, e capire quale
            mlflow.prophet.log_model(pr_model=self.model, artifact_path=artifact_path)
            logger.info(f"### Model logged: {self.model}")

        except Exception as e:
            logger.error(f"### Log model {self.model} failed: {e}")

    def fit(self):

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

    def predict(self, n_days):

        try:
            # configure predictions
            pred_config = self.model.make_future_dataframe(
                periods=n_days, freq="d", include_history=False
            )

            # Add floor and cap
            pred_config["floor"] = self.floor
            pred_config["cap"] = self.cap

            # make predictions
            pred = self.model.predict(pred_config)

            return pred

        except Exception as e:
            logger.error(f"### Predict with model {self.model} failed: {e}")
