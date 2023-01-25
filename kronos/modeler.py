import datetime
import logging
from datetime import timedelta

import mlflow
import numpy as np
import pandas as pd

from kronos.ml_flower import MLFlower
from kronos.models.krns_lumpy import KRNSLumpy
from kronos.models.krns_pmdarima import KRNSPmdarima
from kronos.models.krns_prophet import KRNSProphet

# TODO: Fix del modello tensorflow, per ora è commentato perchè non riuscivo a farlo eseguire in locale
from kronos.models.krns_tensorflow import KRNSTensorflow

logger = logging.getLogger(__name__)


class Modeler:
    """
    Class to manage all modelling activities.
    """

    # TODO: fcst_competition_metric_weights è associato alle metriche su base posizionale, sarebbe meglio su base del nome, tipo un dizionario

    def __init__(
        self,
        ml_flower: MLFlower,
        data: pd.DataFrame,
        key_col: str,
        date_col: str,
        metric_col: str,
        fcst_col: str,
        models_config: dict,
        current_date: datetime.date,
        fcst_first_date: datetime.date,
        n_test: int,
        n_unit_test: int,
        fcst_horizon: int,
        horizon: int,
        dt_creation_col: str,
        dt_reference_col: str,
        fcst_competition_metrics: list,
        fcst_competition_metric_weights: list,
        future_only: bool,
        x_reg_columns: list,
    ) -> None:

        """
        Initialization method.

        Other than init attributes, the *key_code* and *max_value* attributes are extracted from data: the first is used to name different objects (e.g. name of the mlflow experiment); the second to set a cap in the training of certain models.

        :param MLFlower ml_flower: The MLFlower instance used to interact with mlflow.
        :param pd.DataFrame data: The pandas DataFrame containing all the information.
        :param str key_col: The name of the column indicating the time series key.
        :param str date_col: The name of the column indicating the time dimension.
        :param str metric_col: The name of the column indicating the dimension to forecast.
        :param str fcst_col: The name of the column indication the forecast.
        :param dict models_config: The dict containing all the model configurations.
        :param datetime.date current_date: Current processing date.
        :param datetime.date fcst_first_date: Date of first day of forecast, usually is the day following the current date.
        :param int n_test: Number of observations to use as test set.
        :param int n_unit_test: Number of points to forecast in the model deployment's unit test.
        :param int fcst_horizon: Number of points to forecast.
        :param str dt_creation_col: The name of the column indicating the forecast creation date.
        :param str dt_reference_col: The name of the column indicating the date used as reference date for forecast.
        :param list fcst_competition_metrics: List of metrics to be used in the competition.
        :param list fcst_competition_metric_weights: List of weights for metrics to be used in the competition.
        :param bool future_only: Whether to return predicted missing values between the last observed date and the forecast first date (*False*) or only future values (*True*), i.e. those from the forecast first date onwards.
        :param list x_reg_columns: List od exogenous regressor columns.

        :return: No return.

        **Example**

        .. code-block:: python

            modeler = Modeler(
                    client=MLFlower(client=client),
                    data=df,
                    key_col='id',
                    date_col='date',
                    metric_col='y',
                    fcst_col='y_hat',
                    models_config={"pmdarima_1":{"model_flavor":"pmdarima","m":7,"seasonal":true}},
                    current_date=(date.today() + timedelta(-1)).strftime('%Y-%m-%d'),
                    fcst_first_date=date.today().strftime('%Y-%m-%d'),
                    n_test=7,
                    n_unit_test=7,
                    fcst_horizon=7,
                    dt_creation_col='creation_date',
                    dt_reference_col='reference_date',
                    fcst_competition_metrics=['rmse', 'mape'],
                    fcst_competition_metric_weights=[0.5, 0.5],
                    future_only=True
                )

        """

        # Input attributes
        self.ml_flower = ml_flower
        self.data = data
        self.key_col = key_col
        self.date_col = date_col
        self.metric_col = metric_col
        self.fcst_col = fcst_col
        self.models_config = models_config
        self.current_date = current_date
        self.fcst_first_date = fcst_first_date
        self.n_test = n_test
        self.n_unit_test = n_unit_test
        self.fcst_horizon = fcst_horizon
        self.horizon = horizon
        self.dt_creation_col = dt_creation_col
        self.dt_reference_col = dt_reference_col
        self.fcst_competition_metrics = fcst_competition_metrics
        self.fcst_competition_metric_weights = fcst_competition_metric_weights
        self.future_only = future_only
        self.x_reg_columns = x_reg_columns

        # Defined attributes
        self.key_code = str(self.data[self.key_col].iloc[0])
        self.max_value = self.data[self.metric_col].max()

        # Empty/placeholder attributes
        self.train_data = None
        self.test_data = None
        self.pred_data = None

        self.df_performances = pd.DataFrame(
            columns=["model_name", "model_config", "model", "run_id"]
            + self.fcst_competition_metrics
        ).set_index("model_name")
        self.winning_model_name = None
        self.variables = None

    @staticmethod
    def evaluate_model(actual: np.ndarray, pred: np.ndarray, metrics: list) -> dict:
        """
        A static method which computes a list of metrics with which to evaluate the model's predictions.
        Currently supported metrics are: rmse, mape.

        :param np.ndarray actual: Array containing the actual data.
        :param np.ndarray pred: Array containing the predicted data.
        :param list metrics: List of metrics (as strings) to compute.

        :return: *(dict)* A dictionary containing all metric names as keys and their computed values as values.
        """

        try:
            supported_metrics = [
                "rmse",
                "mape",
                "max_perc_diff",
                "max_perc_diff_3_days",
            ]
            out = {}

            for metric in metrics:

                logger.debug(f"### Performing evaluation using {metric} metric.")

                # Transform metric in lower case and remove whitespaces
                metric = metric.lower().replace(" ", "")

                if metric not in supported_metrics:
                    logger.error(
                        f"### Requested metric {metric} is not supported. Available metrics are: {supported_metrics}"
                    )
                else:
                    if metric == "rmse":
                        value = ((actual - pred) ** 2).mean() ** 0.5
                    elif metric == "mape":
                        value = (
                            np.abs((actual - pred) / (actual + 0.0001))
                        ).mean() * 100
                    elif metric == "max_perc_diff":
                        # identifico l'osservazione col massimo scarto
                        idx_max = np.argmax(np.abs(actual))
                        # estraggo actual e pred relativi e calcolo il delta perc
                        act_val = actual[idx_max]
                        pred_val = pred[idx_max]
                        value = abs(((act_val - pred_val) / (act_val + 0.0001)) * 100)
                    elif metric == "max_perc_diff_3_days":
                        # identifico l'osservazione col massimo scarto
                        actual_3 = actual[:3]
                        pred_3 = pred[:3]

                        idx_max = np.argmax(np.abs(actual_3))
                        # estraggo actual e pred relativi e calcolo il delta perc
                        act_val = actual_3[idx_max]
                        pred_val = pred_3[idx_max]
                        value = abs(((act_val - pred_val) / (act_val + 0.0001)) * 100)
                    else:
                        value = np.Inf

                    out[metric] = value
                    logger.debug(
                        f"### Evaluation on {metric} completed with value {value}."
                    )

            return out

        except Exception as e:
            logger.error(f"### Model evaluation with metrics {metrics}: {e}")

    def train_test_split(self) -> None:
        """
        A method to split data into train/test splits.

        :return: No return.
        """

        try:
            logger.debug("### Performing train/test split.")

            if self.data.shape[0] - self.n_test < self.n_test:
                logger.warning(
                    f"### Not enough records to perform train/test split: {self.data.shape[0]} rows, {self.n_test} for test"
                )
                raise Exception(
                    f"### Not enough records to perform train/test split: {self.data.shape[0]} rows, {self.n_test} for test"
                )
            else:
                self.train_data = self.data.sort_values(
                    by=[self.date_col], ascending=False
                ).iloc[self.n_test + self.fcst_horizon :, :]

                self.test_data = self.data.sort_values(
                    by=[self.date_col], ascending=False
                ).iloc[self.n_test : self.n_test + self.fcst_horizon, :]

                self.pred_data = self.data.sort_values(
                    by=[self.date_col], ascending=False
                ).iloc[: self.fcst_horizon, :]
                logger.debug("### Train/test split completed.")

        except Exception as e:
            logger.error(f"### Train test split failed: {e}")

    def training(self) -> None:
        """
        A method to perform all the training activities:

            1. Define an mlflow experiment.
            2. Perform train/test split of data.
            3. Init all kronos models to train and store them in a dictionary.
            4. For each model to train:

                1. Start an mlflow run.
                2. Preprocess data (*e.g. renaming columns in prophet models*).
                3. Log model params in the mlflow run.
                4. Fit the model.
                5. Log the fitted model as artifact in the mlflow run.
                6. Compute evaluation metrics and add them in a *performance dataframe* (to later compare models).
                7. Log metrics in the mlflow run.
                8. End run.

        :return: No return.
        """

        try:
            # Create experiment
            experiment_path = f"/mlflow/experiments/{self.key_code}"
            self.ml_flower.get_experiment(experiment_path=experiment_path)

            # Train/Test split
            self.train_test_split()

            # Init all models
            models = self.create_all_models(models_config=self.models_config)

            for model_name, model in models.items():
                try:
                    # Start run
                    run_name = datetime.datetime.utcnow().isoformat()
                    run = self.ml_flower.start_run(run_name=run_name)

                    # Store run id
                    run_id = run.info.run_id

                    # Preprocess
                    model.preprocess()

                    # Log model params
                    # model.log_params(client=self.ml_flower.client, run_id=run_id)

                    # Fit the model
                    model.fit()

                    # Log the model
                    # model.log_model(artifact_path="model")

                    # Make predictions
                    test_data_first_date = self.test_data[self.date_col].min()
                    pred = model.predict(
                        # n_days=self.n_test,
                        n_days=self.fcst_horizon,
                        fcst_first_date=test_data_first_date,
                        future_only=True,
                        test=True,
                        return_conf_int=True,
                    )

                    if model_name.split("_")[0].lower() == "pmdarima":
                        pred_method = model.PREDICTION_METHODS
                    else:
                        pred_method = [""]

                    for _pred_method in pred_method:
                        if model_name.split("_")[0].lower() == "pmdarima":
                            model.pred_method = _pred_method
                            if not mlflow.active_run():
                                # Start run
                                run_name = datetime.datetime.utcnow().isoformat()
                                run = self.ml_flower.start_run(run_name=run_name)
                                run_id = run.info.run_id

                        model.log_params(client=self.ml_flower.client, run_id=run_id)
                        model.log_model(artifact_path="model")
                        # Compute rmse and mape
                        train_evals = self.evaluate_model(
                            actual=self.test_data.sort_values(
                                by=[self.date_col], ascending=True
                            )[self.fcst_horizon - self.horizon :][
                                self.metric_col
                            ].values,
                            pred=pred.sort_values(by=[self.date_col], ascending=True)[
                                self.fcst_horizon - self.horizon :
                            ][self.fcst_col + _pred_method].values,
                            metrics=self.fcst_competition_metrics,
                        )

                        # TODO: DF Performance si popola in modo posizionale, sarebbe meglio se si popolasse
                        #  tramite un dizionario mantenendo comunque l'indice
                        self.df_performances.loc[model_name + _pred_method] = [
                            self.models_config[model_name],
                            model,
                            run_id,
                        ] + list(train_evals.values())

                        for key, val in train_evals.items():
                            self.ml_flower.client.log_metric(run_id, key, val)

                        self.ml_flower.end_run()

                except Exception as e:
                    logger.error(f"### Model {model_name} training failed: {e}")
                    self.ml_flower.end_run()

        except Exception as e:
            logger.error(f"### Training failed: {e}")

    def prod_model_eval(self) -> None:
        """
        Method used to retrieve the current production model from mlflow Model Registry.
        The model is used to predict on test set and compute evaluation metrics.

        :return: No return.
        """

        try:
            # Retrieve the model
            model, flavor = self.ml_flower.load_model(
                model_uri=f"models:/{self.key_code}/Production"
            )

            krns_model = self.model_generation(
                model_flavor=flavor, model_config={}, trained_model=model
            )

            # Predict with current production model (on test set)
            test_data_first_date = (
                self.test_data[self.date_col].sort_values(ascending=True).iloc[0]
            )
            pred = krns_model.predict(
                n_days=self.fcst_horizon,
                fcst_first_date=test_data_first_date,
                future_only=True,
                test=True,
                return_conf_int=True,
            )

            # Compute rmse
            prod_evals = self.evaluate_model(
                actual=self.test_data.sort_values(by=[self.date_col], ascending=True)[
                    self.fcst_horizon - self.horizon :
                ][self.metric_col].values,
                pred=pred.sort_values(by=[self.date_col], ascending=True)[
                    self.fcst_horizon - self.horizon :
                ][self.fcst_col].values,
                metrics=self.fcst_competition_metrics,
            )
            self.df_performances.loc["prod_model"] = [
                None,
                krns_model,
                None,
            ] + list(prod_evals.values())

        except Exception as e:
            logger.warning(f"### Prod model prediction failed: {e}")

    def competition(self) -> None:
        """
        Method used to find the best available model among those trained and the one currently in production.
        The weighted average of all metric errors is computed: the model associated with the minimum value is the winning one.

        :return: No return.
        """

        # Compute average error and compare
        try:
            # Standardize all metric columns
            for metric in self.fcst_competition_metrics:
                mean, std = (
                    self.df_performances[metric].mean(),
                    self.df_performances[metric].std(),
                )
                self.df_performances[metric] = self.df_performances[metric].apply(
                    lambda x: (x - mean) / std
                )

            # Compute weighted average of all metrics
            self.df_performances["error_avg"] = self.df_performances.apply(
                lambda x: (
                    (
                        (
                            np.array(
                                [
                                    x[_metric]
                                    for _metric in self.fcst_competition_metrics
                                ]
                            )
                            * np.array(self.fcst_competition_metric_weights)
                        ).sum()
                    )
                    / np.array(self.fcst_competition_metric_weights).sum()
                ),
                axis=1,
            )

            # Find winning model, i.e. the one with the minimum error_avg
            self.winning_model_name = self.df_performances.iloc[
                self.df_performances["error_avg"].argmin()
            ].name

        except Exception as e:
            logger.error(f"### Competition failed: {e}")

    def unit_test(self, model_version_name: str, model_version_stage: str) -> str:
        """
        Method to perform an infrastructure unit test on a mlflow model.
        The model is first retrieved from the mlflow Model Registry, then it is used to predict n data points.
        If the number of output predictions matches the number of predictions required then the test result is *OK*, otherwise *KO*.

        :param str model_version_name: The name of the model to test.
        :param str model_version_stage: The stage of the model to test.

        :return: *(str)* The unit test status: 'OK' or 'KO'.
        """
        try:
            logger.info("### Performing model unit test")

            # Retrieve the model
            model, flavor = self.ml_flower.load_model(
                model_uri=f"models:/{model_version_name}/{model_version_stage}"
            )

            krns_model = self.model_generation(
                model_flavor=flavor, model_config={}, trained_model=model
            )

            # Predict with the model
            unit_test_fcst_first_date = self.current_date + datetime.timedelta(days=1)
            pred = krns_model.predict(
                n_days=self.n_unit_test,
                fcst_first_date=unit_test_fcst_first_date,
                future_only=True,
                test=True,
                return_conf_int=True,
            )

            # Check quality
            unit_test_status = "OK" if len(pred) == self.n_unit_test else "KO"
            logger.info(
                f"### Unit test result: {unit_test_status} - requested: {self.n_unit_test} - predicted: {len(pred)}"
            )

            return unit_test_status

        except Exception as e:
            logger.error(
                f"### Unit test of model {model_version_name} in stage {model_version_stage} failed: {e}"
            )

    def deploy(self) -> None:
        """
        Method to perform the winning model deploy into the mlflow Model Registry.
        The main steps are:

            1. Register the mlflow run artifact model into the mlflow Model Registry.
            2. Set the model flavor as tag of the model, for information purposes only (not required by mlflow).
            3. Promote the model to the "Staging" stage of the mlflow Model Registry.
            4. Perform the model unit test.
            5. If the unit test succeed, promote the model to the "Production" stage of the mlflow Model Registry.

        :return: No return.
        """
        try:
            # Get winning model run id
            winning_model_run_id = self.df_performances.loc[self.winning_model_name][
                "run_id"
            ]
            winning_model_config = self.df_performances.loc[self.winning_model_name][
                "model_config"
            ]

            # Register the model
            logger.info("### Registering the model")
            model_uri = f"runs:/{winning_model_run_id}/model"

            model_details = self.ml_flower.register_model(
                model_uri=model_uri, model_name=self.key_code, timeout_s=10
            )

            # Set model flavor tag
            if "model_flavor" in winning_model_config:
                logger.info("### Setting model flavor tag")
                self.ml_flower.set_model_tag(
                    model_version_details=model_details,
                    tag_key="model_flavor",
                    tag_value=winning_model_config.get("model_flavor", ""),
                )

            # Transition to "staging" stage and archive the last one (if present)
            model_version = self.ml_flower.promote_model(
                model_version_details=model_details,
                stage="staging",
                archive_existing_versions=True,
            )

            # Unit test the model
            logger.info("### Performing model unit test")
            unit_test_status = self.unit_test(
                model_version_name=model_version.name,
                model_version_stage=model_version.current_stage,
            )

            if unit_test_status == "OK":
                # Take the current staging model and promote it to production
                # Archive the "already in production" model
                # Return status code
                logger.info("### Deploying model to Production.")
                model_version = self.ml_flower.promote_model(
                    model_version_details=model_version,
                    stage="production",
                    archive_existing_versions=True,
                )

                deploy_result = "OK" if model_version.status == "READY" else "KO"
                logger.info(f"### Deploy result: {deploy_result}")

        except Exception as e:
            logger.error(
                f"### Deployment of model {self.winning_model_name} failed: {e}"
            )

    def prediction(self) -> pd.DataFrame:
        """
        Method to perform prediction using an mlflow model.
        The model is retrieved from the "Production" stage of the mlflow Model Registry, then it is used to provide the predictions.
        Finally, informative columns are added (e.g. days from last execution) and negative predictions are removed.

        :return: *(pd.DataFrame)* Pandas DataFrame containing the predictions.
        """

        try:
            # Retrieve production model
            model, flavor = self.ml_flower.load_model(
                model_uri=f"models:/{self.key_code}/Production"
            )

            krns_model = self.model_generation(
                model_flavor=flavor, model_config={}, trained_model=model
            )

            # Get predictions
            pred = krns_model.predict(
                fcst_first_date=self.fcst_first_date,
                n_days=self.fcst_horizon,
                future_only=self.future_only,
                test=False,
                return_conf_int=True,
            )

            # Keep only relevant columns
            pred = pred[[self.date_col, self.fcst_col]]

        except Exception as e:
            logger.error(f"### Prediction failed: {e}")

            # Keep only historic data
            historic_data = self.data[self.data[self.date_col] < self.fcst_first_date]

            # Compute last observed historical day
            last_observed_day = historic_data[self.date_col].max()

            # Compute the difference between last_observed_day and fcst_first_date
            difference = (self.fcst_first_date - last_observed_day).days

            # Compute actual forecast horizon
            fcst_horizon = difference + self.fcst_horizon - 1

            # Get last value
            last_value = historic_data.sort_values(
                by=self.date_col, ascending=False, inplace=False
            ).iloc[0][self.metric_col]

            # Create dummy pred df
            pred = pd.DataFrame(
                {
                    self.date_col: [
                        last_observed_day + datetime.timedelta(days=x)
                        for x in range(1, fcst_horizon + 1)
                    ],
                    self.fcst_col: [last_value for x in range(fcst_horizon)],
                }
            )

            # Keep relevant data
            if self.future_only:
                pred = pred[pred[self.date_col] >= self.fcst_first_date]

        # Add to predictions
        pred[self.dt_reference_col] = datetime.date.today() + timedelta(days=1)
        pred[self.dt_creation_col] = datetime.date.today()

        # Add key code to predictions
        pred[self.key_col] = self.key_code

        # Remove negative values
        pred[self.fcst_col] = pred[self.fcst_col].apply(lambda x: x if x >= 0 else 0)

        return pred

    def create_all_models(self, models_config: dict) -> dict:
        """
        Method to instantiate all the kronos models.

        :param dict models_config: The dict containing all the model configurations.

        :return: *(dict)* The list with all the instances of kronos models.
        """

        try:
            # For each model config create its instance
            models = {}
            for model_name, model_config in models_config.items():
                model_flavor = model_name.split("_")[0]
                model = self.model_generation(
                    model_flavor=model_flavor,
                    model_config=model_config,
                    trained_model=None,
                )
                # Add model to the model list
                models[model_name] = model
            return models

        except Exception as e:
            logger.error(
                f"### Create all models failed: {e} - models config are: {models_config}"
            )

    def model_generation(
        self, model_flavor: str, model_config: dict, trained_model: None
    ):
        """
        Method to instantiate a single kronos models.

        :param str model_flavor: The flavor of the model (e.g. 'prophet').
        :param dict model_config: The dict containing all the model configurations.
        :param trained_model: An already fitted model. To instantiate a kronos model from an already fitted model.

        :return: The kronos model instantiated.
        """
        try:
            if model_flavor == "prophet":
                model = KRNSProphet(
                    modeler=self,
                    interval_width=model_config.get("interval_width", 0.95),
                    growth=model_config.get("growth", "linear"),
                    daily_seasonality=model_config.get("daily_seasonality", False),
                    weekly_seasonality=model_config.get("weekly_seasonality", True),
                    yearly_seasonality=model_config.get("yearly_seasonality", True),
                    seasonality_mode=model_config.get(
                        "seasonality_mode", "multiplicative"
                    ),
                    floor=model_config.get("floor", None),
                    cap=self.max_value,
                    country_holidays=model_config.get("country_holidays", "IT"),
                    model=trained_model,
                )
            elif model_flavor == "pmdarima":
                if trained_model:
                    pred_method = trained_model[1]
                    trained_model = trained_model[0]
                else:
                    pred_method = None
                model = KRNSPmdarima(
                    modeler=self,
                    m=model_config.get("m", 7),
                    seasonal=model_config.get("seasonal", True),
                    model=trained_model,
                    select_variables=model_config.get("select_variables", True),
                )
                model.pred_method = pred_method
            elif model_flavor == "lumpy":
                model = KRNSLumpy(
                    modeler=self,
                    m=model_config.get("m", 1),
                    start_P=model_config.get("start_P", 1),
                    max_P=model_config.get("max_P", 1),
                    start_D=model_config.get("start_D", 0),
                    max_D=model_config.get("max_D", 0),
                    start_Q=model_config.get("start_Q", 1),
                    max_Q=model_config.get("max_Q", 1),
                    start_p=model_config.get("start_p", 1),
                    max_p=model_config.get("max_p", 1),
                    start_d=model_config.get("start_d", 0),
                    max_d=model_config.get("max_d", 1),
                    start_q=model_config.get("start_q", 1),
                    max_q=model_config.get("max_q", 1),
                    model=trained_model,
                )
            # TODO: sbloccare tensorflow
            elif model_flavor == "tensorflow":
                model = KRNSTensorflow(
                    modeler=self,
                    nn_type=model_config.get("nn_type", "rnn"),
                    n_units=model_config.get("n_units", 128),
                    activation=model_config.get("activation", "relu"),
                    epochs=model_config.get("epochs", 25),
                    n_inputs=model_config.get("n_inputs", 30),
                    model=trained_model,
                )
            else:
                raise ValueError(f"Model {model_flavor} not supported.")

            return model

        except Exception as e:
            logger.error(
                f"### Model generation of model flavor {model_flavor} with model config {model_config} and trained model {trained_model} failed: {e}"
            )
