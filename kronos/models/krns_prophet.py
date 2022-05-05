from prophet import Prophet
import pandas as pd
import logging

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
            train_data: pd.DataFrame = None,
            test_data: pd.DataFrame = None,
            interval_width: float = 0.95,
            growth: str = 'linear',
            daily_seasonality: bool = True,
            weekly_seasonality: bool = True,
            yearly_seasonality: bool = True,
            seasonality_mode: str = 'multiplicative',
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
        self.daily_seasonality = daily_seasonality if not model else model.daily_seasonality
        self.weekly_seasonality = weekly_seasonality if not model else model.weekly_seasonality
        self.yearly_seasonality = yearly_seasonality if not model else model.yearly_seasonality
        self.seasonality_mode = seasonality_mode if not model else model.seasonality_mode

        # To load an already configured model
        self.model = model

        self.model_params = {
            "interval_width": self.interval_width,
            "growth": self.growth,
            "daily_seasonality": self.daily_seasonality,
            "weekly_seasonality": self.weekly_seasonality,
            "yearly_seasonality": self.yearly_seasonality,
            "seasonality_mode": self.seasonality_mode
        }

    def preprocess(self):
        """
        Get the dataframe into the condition to be processed by the model.
        :return: No return.
        """
        if self.train_data:
            self.train_data.rename(columns={self.date_col: "ds", self.metric_col: "y"}, inplace=True)
        else:
            print("No training data")

        if self.test_data:
            self.test_data.rename(columns={self.date_col: "ds", self.metric_col: "y"}, inplace=True)
        else:
            print("No test data")

    def fit(self):
        if self.train_data:
            # Define the model
            self.model = Prophet(
                interval_width=self.interval_width,
                growth=self.growth,
                daily_seasonality=self.daily_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                yearly_seasonality=self.yearly_seasonality,
                seasonality_mode=self.seasonality_mode
            )

            # Fit the model
            self.model.fit(self.train_data)

        else:
            print("No training data")

    def predict(self, n_days):

        if self.model:
            # configure predictions
            pred_config = self.model.make_future_dataframe(periods=n_days, freq='d', include_history=False)

            # make predictions
            pred = self.model.predict(pred_config)

            return pred

        else:
            print("Trained model not present")
