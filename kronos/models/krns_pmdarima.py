import copy
import datetime
import logging
from datetime import timedelta

import mlflow
import numpy as np
import pandas as pd
import pmdarima as pm
import scipy.stats as sps
from mlflow.tracking import MlflowClient
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)


class KRNSPmdarima:
    """
    Class to implement pm.arima.arima.ARIMA in kronos.
    """

    PREDICTION_METHODS = ["percentile_85", "Confidence_intervall"]

    def __init__(
        self,
        modeler,  # TODO: How to explicit its data type without incur in [...] most likely due to a circular import
        model: pm.arima.arima.ARIMA = None,
        m: int = 7,
        seasonal: bool = True,
        select_variables: bool = True,
        variables: list = None,
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
        self.select_variables = select_variables
        self._pred_method = None
        self.variables = variables

        # To load an already configured model
        self.model = model

        self.model_params = {
            "m": self.m,
            "seasonal": self.seasonal,
            "select_variables": self.select_variables,
        }

    @property
    def pred_method(self):
        return self._pred_method

    @pred_method.setter
    def pred_method(self, pred_meth):
        self._pred_method = pred_meth
        self.model_params["prediction_method"] = pred_meth

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
            # Find meaningful variable using linear regression
            if self.select_variables:
                rfe = RFE(estimator=LinearRegression())

                rfe.fit(
                    self.modeler.train_data.drop(self.modeler.metric_col, axis=1),
                    self.modeler.train_data[self.modeler.metric_col],
                )

                self.variables = list()

                for i in range(
                    self.modeler.train_data.drop(self.modeler.metric_col, axis=1).shape[
                        1
                    ]
                ):
                    if rfe.support_[i]:
                        self.variables.append(
                            self.modeler.train_data.drop(
                                self.modeler.metric_col, axis=1
                            ).columns[i]
                        )
            else:
                self.variables = list(
                    self.modeler.train_data.drop(
                        self.modeler.metric_col, axis=1
                    ).columns
                )
            train_variables = self.modeler.train_data[self.variables]
            train_variables = train_variables.apply(pd.to_numeric)

            # Define the model
            self.model = pm.auto_arima(
                y=self.modeler.train_data.loc[:, self.modeler.metric_col],
                exogenous=train_variables,
                seasonal=self.seasonal,
                m=self.m,
                sarimax_kwargs={
                    "enforce_stationarity": False,
                },
            )

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
                self.variables = self.model.arima_res_.model.exog_names

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
                        self.variables = self.model.arima_res_.model.exog_names

                    exogenous = self.modeler.test_data.sort_index()[self.variables]
                else:
                    exogenous = self.modeler.pred_data.set_index(
                        [self.modeler.date_col]
                    ).sort_index()[self.variables]

                # TODO: return_conf_int a True serve per considerare il 75esimo percentile della distribuzione prevista al posto della media--> impostare l'85esimo
                #       è necessario modificare la funzione di predict di pmdarima
                prediction = self.model.predict(
                    n_periods=fcst_horizon,
                    exogenous=exogenous,
                    return_conf_int=True,
                    alpha=0.05,
                )

                # is taken from the pmdarima.arima.Arima.predict()
                # has been skiped some checks of the original implementation

                arima = self.model.arima_res_
                end = arima.nobs + fcst_horizon - 1
                results = arima.get_prediction(
                    start=arima.nobs, end=end, exog=exogenous
                )

                if isinstance(results.predicted_mean, pd.core.series.Series):
                    # the results of get prediction (SARIMAX) puo essere suia pandas sia numpy
                    mean = results.predicted_mean.to_numpy()
                    variance = results.var_pred_mean.to_numpy()
                else:
                    mean = results.predicted_mean
                    variance = results.var_pred_mean
                    # compute the normal distribution for each prediction
                    # use associated variances and mean
                dists = [
                    sps.norm(loc=m, scale=s)
                    for m, s in zip(
                        np.array(mean),
                        np.sqrt(variance),
                    )
                ]
                # Compute the 85 percentile of the normal distribution
                prediction1 = [i.ppf(0.85) for i in dists]
                # logger.info(f"### Predict with percentile 85")

                prediction2 = self.model.predict(
                    n_periods=fcst_horizon,
                    exogenous=exogenous,
                    return_conf_int=True,
                )
                prediction2 = [fcst[1] for fcst in prediction[1]]
                # logger.info(f"### Predict with Confidnce intervall")

                if not self.pred_method:
                    pred = pd.DataFrame(
                        data={
                            self.modeler.date_col: [
                                last_training_day + datetime.timedelta(days=x)
                                for x in range(1, fcst_horizon + 1)
                            ],
                            self.modeler.fcst_col
                            + self.PREDICTION_METHODS[0]: prediction1,
                            self.modeler.fcst_col
                            + self.PREDICTION_METHODS[1]: prediction2,
                        }
                    )
                elif self.pred_method == self.PREDICTION_METHODS[0]:
                    pred = pd.DataFrame(
                        data={
                            self.modeler.date_col: [
                                last_training_day + datetime.timedelta(days=x)
                                for x in range(1, fcst_horizon + 1)
                            ],
                            self.modeler.fcst_col: prediction1,
                        }
                    )
                elif self.pred_method == self.PREDICTION_METHODS[1]:
                    pred = pd.DataFrame(
                        data={
                            self.modeler.date_col: [
                                last_training_day + datetime.timedelta(days=x)
                                for x in range(1, fcst_horizon + 1)
                            ],
                            self.modeler.fcst_col: prediction2,
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

            return pred

        except Exception as e:
            logger.error(f"### Predict with model {self.model} failed: {e}")
