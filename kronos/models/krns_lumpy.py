import copy
import datetime
import logging
from datetime import timedelta

import mlflow
import numpy as np
import pandas as pd
import pmdarima as pm
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class KRNSLumpy:
    """
    Class to implement a specific forecats model for lumpy class in kronos.
    """

    def __init__(
        self,
        modeler,
        model: pm.arima.arima.ARIMA = None,
        variables: list = None,
        start_P: int = 1,
        max_P: int = 1,
        start_D: int = 0,
        max_D: int = 0,
        start_Q: int = 1,
        max_Q: int = 1,
        m: int = 1,
        start_p: int = 1,
        max_p: int = 1,
        start_d: int = 0,
        d: int = 0,
        max_d: int = 0,
        start_q: int = 1,
        max_q: int = 1,
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

        # To load an already configured model
        self.model = model

        self.start_P = start_P
        self.max_P = max_P
        self.start_D = start_D
        self.max_D = max_D
        self.start_Q = start_Q
        self.max_Q = max_Q
        self.m = m
        self.start_p = start_p
        self.max_p = max_p
        self.start_d = start_d
        self.d = self.start_d
        #self.max_d = max_d
        self.start_q = start_q
        self.max_q = max_q

        self.model_params = {
            "start_P": self.start_P,
            "max_P": self.max_P,
            "start_D": self.start_D,
            "max_D": self.max_D,
            "start_Q": self.start_Q,
            "max_Q": self.max_Q,
            "m": self.m,
            "start_p": self.start_p,
            "max_p": self.max_p,
            #"start_d": self.start_d,
            #"max_d": self.max_d,
            "d": self.d,
            "start_q": self.start_q,
            "max_q": self.max_q,
        }

        self.variables = variables

    def preprocess(self) -> None:
        """
        Get the dataframe into the condition to be processed by the model: keep only the *date* and *metric* column.

        :return: No return.
        """

        try:
            self.modeler.data.drop(
                self.modeler.data.columns.difference(
                    [self.modeler.date_col, self.modeler.metric_col]
                    + self.modeler.x_reg_columns
                ),
                axis=1,
                inplace=True,
            )
            if self.modeler.data.index.name != self.modeler.date_col:
                self.modeler.data.set_index(self.modeler.date_col, inplace=True)

            self.modeler.data["on_off"] = np.where(
                self.modeler.data[self.modeler.metric_col] > 0, 1, 0
            )

        except Exception as e:
            logger.warning(
                f"### Preprocess data failed: {e} - {self.modeler.data.head(1)}"
            )

        try:
            self.modeler.train_data.dropna(
                subset=[self.modeler.metric_col],
                inplace=True,
            )

            self.modeler.train_data.drop(
                self.modeler.train_data.columns.difference(
                    [self.modeler.date_col, self.modeler.metric_col]
                    + self.modeler.x_reg_columns
                ),
                axis=1,
                inplace=True,
            )

            if self.modeler.train_data.index.name != self.modeler.date_col:
                self.modeler.train_data.set_index(self.modeler.date_col, inplace=True)

            self.modeler.train_data["on_off"] = np.where(
                self.modeler.train_data[self.modeler.metric_col] > 0, 1, 0
            )

        except Exception as e:
            logger.warning(
                f"### Preprocess train data failed: {e} - {self.modeler.train_data.head(1)}"
            )

        try:
            self.modeler.test_data.drop(
                self.modeler.test_data.columns.difference(
                    [self.modeler.date_col, self.modeler.metric_col]
                    + self.modeler.x_reg_columns
                ),
                axis=1,
                inplace=True,
            )

            if self.modeler.test_data.index.name != self.modeler.date_col:
                self.modeler.test_data.set_index(self.modeler.date_col, inplace=True)

            self.modeler.test_data["on_off"] = np.where(
                self.modeler.test_data[self.modeler.metric_col] > 0, 1, 0
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
        try:
            # Define the model
            # self.model = pm.ARIMA(
            #     (self.max_p, self.max_d, self.max_q),
            #     seasonal_order=(self.max_P, self.max_D, self.max_Q, self.m),
            # )
            # self.model.fit(
            #     y=self.modeler.train_data.loc[:, self.modeler.metric_col],
            # )
            if len(self.modeler.x_reg_columns)<1:
                exogenous = None
            else:
                exogenous = self.modeler.train_data[self.modeler.x_reg_columns]
                exogenous = exogenous.apply(pd.to_numeric, errors='coerce')
            self.model = pm.auto_arima(
                y=self.modeler.train_data.loc[:, self.modeler.metric_col],
                exogenous=exogenous,
                start_P=self.start_P,
                max_P=self.max_P,
                start_D=self.start_D,
                max_D=self.max_D,
                start_Q=self.start_Q,
                max_Q=self.max_Q,
                m=self.m,
                start_p=self.start_p,
                max_p=self.max_p,
                d=self.start_d,
                #max_d=self.max_d,
                start_q=self.start_q,
                max_q=self.max_q,
                seasonal=True,
            )
            
            if type(self.modeler.train_data.index[0]) != datetime.date:
                self.preprocess()
            # Add last training day attribute
            self.model.last_training_day = self.modeler.train_data.index.max()

        except Exception as e:
            logger.error(
                f"### Fit with model {self.model} failed: {e} - on data {self.modeler.train_data.head(1)}"
            )

    def predict(
        self,
        n_days: int,
        fcst_first_date: datetime.date = datetime.date.today(),
        future_only: bool = True,
        test: bool = False,
        return_conf_int: bool = True,
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
            if type(self.modeler.data.index[0]) != datetime.date:
                self.preprocess()

            # Retrieve model last training day
            last_training_day = copy.deepcopy(self.model.last_training_day)

            # Update model with last data (if any) and update last_training_day value
            update_data = self.modeler.data[
                (last_training_day < self.modeler.data.index)
                & (self.modeler.data.index < fcst_first_date)
            ]
            if len(update_data) > 0:
                self.variables = self.modeler.x_reg_columns

                self.model.update(
                    y=update_data[self.modeler.metric_col].to_numpy(),
                    exogenous=update_data[self.variables],
                )
                last_training_day = update_data.index.max()

            # Compute the difference between last_training_day and fcst_first_date
            difference = (fcst_first_date - last_training_day).days

            # Compute actual forecast horizon
            fcst_horizon = max(difference + n_days - 1, 0)

            # Make predictions
            if fcst_horizon > 0:
                if test:
                    if self.variables is None:
                        self.variables = self.modeler.x_reg_columns

                    exogenous = self.modeler.test_data.sort_index()[self.variables]
                else:
                    exogenous = self.modeler.pred_data.set_index(
                        [self.modeler.date_col]
                    ).sort_index()[self.variables]

                prediction = self.model.predict(
                    n_periods=fcst_horizon,
                    exogenous=exogenous,
                    return_conf_int=True,
                )

                pred = pd.DataFrame(
                    data={
                        self.modeler.date_col: [
                            last_training_day + datetime.timedelta(days=x)
                            for x in range(1, fcst_horizon + 1)
                        ],
                        self.modeler.fcst_col: [fcst for fcst in prediction[0]],
                    }
                )
            else:
                pred = pd.DataFrame(
                    data={self.modeler.date_col: [], self.modeler.fcst_col: []}
                )

            # Attach actual data on predictions
            # TODO: Capire perchè c'è un controllo sulla difference minore di zero
            if difference < 0:
                # Keep last n actual data (n = difference - 1)
                actual_data = self.modeler.data.sort_index(ascending=True).iloc[
                    difference - 1 :
                ]
                # Reset index
                actual_data.reset_index(inplace=True)
                # Rename columns
                actual_data.rename(
                    columns={self.modeler.metric_col: self.modeler.fcst_col},
                    inplace=True,
                )
                # Concat to pred and reset index
                pred = pd.concat([actual_data, pred])
                pred.reset_index(drop=True, inplace=True)

            # Keep relevant data
            # TODO: capire quando il campo future_only può essere valorizzato a False
            if future_only:
                pred = pred[pred[self.modeler.date_col] >= fcst_first_date]
            if difference < 0:
                pred = pred[
                    pred[self.modeler.date_col]
                    < fcst_first_date + datetime.timedelta(days=n_days)
                ]

            # pred["value_last_week"] = pred.apply(
            #     lambda x: self.modeler.data.loc[x.giorno_gas - timedelta(days=7)][
            #         self.modeler.metric_col
            #     ],
            #     axis=1,
            # )
            # pred[self.modeler.fcst_col] = pred[
            #     [self.modeler.fcst_col, "value_last_week"]
            # ].max(axis=1)
            # pred = pred.drop(["value_last_week"], axis=1)
            
            pred['volume_giorno_fcst'] = self.modeler.test_data.iloc[0]['volume_giorno']

            return pred

        except Exception as e:
            logger.error(f"### Predict with model {self.model} failed: {e}")
