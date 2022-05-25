from kronos.ml_flower import MLFlower
from kronos.models.krns_prophet import KRNSProphet
from kronos.models.krns_pmdarima import KRNSPmdarima
import pandas as pd
import numpy as np
import logging
import datetime

logger = logging.getLogger(__name__)


class Modeler:
    """
    TODO: Doc
    """

    def __init__(
        self,
        ml_flower: MLFlower,
        data: pd.DataFrame,
        key_col: str,
        date_col: str,
        metric_col: str,
        fcst_col: str,
        models_config: dict,
        days_from_last_obs_col: str,
        current_date: datetime.date,
        fcst_first_date: datetime.date,
        n_test: int,
        n_unit_test: int,
        forecast_horizon: int,
        dt_creation_col: str,
        dt_reference_col: str,
        future_only: True,
    ):
        """
        TODO: Doc
        """

        # Input attributes
        self.ml_flower = ml_flower
        self.data = data
        self.key_col = key_col
        self.date_col = date_col
        self.metric_col = metric_col
        self.fcst_col = fcst_col
        self.models_config = models_config
        self.days_from_last_obs_col = days_from_last_obs_col
        self.current_date = current_date
        self.fcst_first_date = fcst_first_date
        self.n_test = n_test
        self.n_unit_test = n_unit_test
        self.forecast_horizon = forecast_horizon
        self.dt_creation_col = dt_creation_col
        self.dt_reference_col = dt_reference_col
        self.future_only = future_only

        # Defined attributes
        self.key_code = str(self.data[self.key_col].iloc[0])
        self.max_value = self.data[self.metric_col].max()

        # Empty attributes
        self.run_id = ""
        self.train_data = pd.DataFrame()
        self.test_data = pd.DataFrame()
        # TODO: Deve essere basato sulle metriche specificate dall'utente
        self.df_performances = pd.DataFrame(
            columns=["model_name", "rmse", "model_config", "model"]
        ).set_index("model_name")
        self.winning_model = ""

    @staticmethod
    def evaluate_model(actual: np.ndarray, pred: np.ndarray, metrics: list):
        """
        # TODO: Doc
        :param actual:
        :param pred:
        :param metrics:
        :return:
        """

        supported_metrics = ["rmse", "mape"]
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
                    value = (np.abs((actual - pred) / actual)).mean() * 100
                else:
                    value = np.Inf

                out[metric] = value
                logger.debug(
                    f"### Evaluation on {metric} completed with value {value}."
                )

        return out

    def train_test_split(self):
        """
        TODO: Doc
        :return:
        """

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
            ).iloc[self.n_test :, :]
            self.test_data = self.data.sort_values(
                by=[self.date_col], ascending=False
            ).iloc[: self.n_test, :]
            logger.debug("### Train/test split completed.")

    def training(self):
        """
        # TODO: Doc
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
                    self.run_id = run.info.run_id

                    # Preprocess
                    model.preprocess()

                    # Log model params
                    model.log_params(client=self.ml_flower.client, run_id=self.run_id)

                    # Fit the model
                    model.fit()

                    # Log the model
                    model.log_model(artifact_path="model")

                    # Make predictions
                    test_data_first_date = self.test_data[self.date_col].sort_values(
                        ascending=True
                    )[0]
                    pred = model.predict(
                        n_days=self.n_test, fcst_first_date=test_data_first_date
                    )

                    # Compute rmse and mape
                    # TODO: Andranno calcolate anche eventuali metriche aggiuntive - lette da una tabella parametrica
                    # TODO: yhat è tipica di prophet o dei nostri krns models, da generalizzare o da inserire come parametro nei krns models
                    train_evals = self.evaluate_model(
                        actual=self.test_data[self.metric_col].values,
                        pred=pred["yhat"].values,
                        metrics=["rmse", "mape"],
                    )

                    # TODO: All'interno di questo df ci dovranno essere n metriche di confronto (definite tramite cockpit dall'utente).
                    #  Il dataframe a questo punto potrebbe essere generato da un dizionario dinamico con le n metriche.
                    self.df_performances.loc[model_name] = [
                        train_evals["rmse"],
                        self.models_config[model_name],
                        model,
                    ]

                    # TODO: Forse da internalizzare a MLFlower
                    for key, val in train_evals.items():
                        self.ml_flower.client.log_metric(self.run_id, key, val)

                    self.ml_flower.end_run()

                except Exception as e:
                    logger.error(f"### Model {model_name} training failed: {e}")
                    self.ml_flower.end_run()

        except Exception as e:
            logger.error(f"### Training failed: {e}")

    def prod_model_eval(self):
        """
        TODO: Doc
        """

        # TODO: Da trasformare inizializzando un model krns prophet o quello che è
        try:
            # Retrieve the model
            prod_model, flavor = self.ml_flower.load_model(
                model_uri=f"models:/{self.key_code}/Production"
            )

            # Predict with current production model (on test set)
            last_prod_model_date = prod_model.history_dates[0].date()
            # TODO: Da rendere agnostico da prophet
            last_test_date = self.test_data.sort_values(
                by="ds", ascending=False, inplace=False
            ).iloc[0]["ds"]
            difference = (last_test_date - last_prod_model_date).days
            pred_config = prod_model.make_future_dataframe(
                periods=difference, freq="d", include_history=False
            )

            # Add floor and cap
            pred_config["floor"] = self.models_config["prophet_1"]["floor"]
            pred_config["cap"] = self.max_value

            pred = prod_model.predict(pred_config)

            # Compute rmse
            # TODO: Andranno calcolate anche eventuali metriche aggiuntive
            prod_rmse = self.evaluate_model(
                actual=self.test_data[self.metric_col].values,
                pred=pred["yhat"].values,
                metrics=["rmse"],
            )
            self.df_performances.loc["prod_model"] = [prod_rmse["rmse"], {}, prod_model]

        except Exception as e:
            logger.warning(f"### Prod model prediction failed: {e}")

    def competition(self):
        """
        # TODO: Doc
        """

        # Compute final score and compare
        # TODO: Dovrà essere la somma di tutte le metriche con cui si vogliono confrontare i modelli
        try:
            self.df_performances["score"] = self.df_performances["rmse"].apply(
                lambda x: -x
            )
            self.winning_model = self.df_performances.iloc[
                self.df_performances["score"].argmax()
            ].name
        except Exception as e:
            logger.error(f"### Competition failed: {e}")

    def deploy(self):
        """
        # TODO: Doc
        """
        try:
            self.ml_flower.deploy(
                run_id=self.run_id,
                key_code=self.key_code,
                models_config=self.df_performances.loc[self.winning_model][
                    "model_config"
                ],
                n_unit_test=self.n_unit_test,
                cap=self.max_value,
            )
        except Exception as e:
            logger.error(f"### Deployment failed: {e}")

    def prediction(self):
        """
        # TODO: Doc
        """

        try:
            # Retrieve production model
            model, flavor = self.ml_flower.load_model(
                model_uri=f"models:/{self.key_code}/Production"
            )

            # Predict with current production model: compute actual forecast horizon needed first
            last_date = model.history_dates[0].date()
            actual_forecast_horizon = (
                (datetime.date.today() + datetime.timedelta(days=self.forecast_horizon))
                - last_date
            ).days
            pred_config = model.make_future_dataframe(
                periods=actual_forecast_horizon, freq="d", include_history=False
            )

            # Add floor and cap
            pred_config["floor"] = self.df_performances.loc[self.winning_model][
                "model_config"
            ].get("floor", 0)
            pred_config["cap"] = self.max_value

            pred = model.predict(pred_config)
            pred = pred[["ds", "yhat"]]

            # Rename from prophet naming
            pred.rename(
                columns={"ds": self.date_col, "yhat": self.fcst_col}, inplace=True
            )

            # Convert to date
            pred[self.date_col] = pred[self.date_col].dt.date

        except Exception as e:
            logger.error(f"### Prediction failed: {e}")

            # Get last value and date
            last_value = self.data.sort_values(
                by=self.date_col, ascending=False, inplace=False
            ).iloc[0][self.metric_col]
            last_date = self.data.sort_values(
                by=self.date_col, ascending=False, inplace=False
            ).iloc[0][self.date_col]

            # Create dummy pred df
            actual_forecast_horizon = (
                (datetime.date.today() + datetime.timedelta(days=self.forecast_horizon))
                - last_date
            ).days
            pred = pd.DataFrame(
                {
                    self.date_col: [
                        last_date + datetime.timedelta(days=x)
                        for x in range(actual_forecast_horizon)
                    ],
                    self.fcst_col: [last_value for i in range(actual_forecast_horizon)],
                }
            )

        # Compute days from last obs, reference date and prediction date
        last_date = self.data.sort_values(
            by=self.date_col, ascending=False, inplace=False
        ).iloc[0][self.date_col]
        days_from_last_obs = (datetime.date.today() - last_date).days
        # Add to predictions
        pred[self.days_from_last_obs_col] = days_from_last_obs
        pred[self.dt_reference_col] = self.fcst_first_date
        pred[self.dt_creation_col] = self.current_date

        # Add key code to predictions
        pred[self.key_col] = self.key_code

        # Keep only future rows
        if self.future_only:
            pred = pred[pred[self.date_col] >= self.fcst_first_date]

        # Flat predictions to floor value
        # TODO: Da rivedere/discutere
        pred[self.fcst_col] = pred[self.fcst_col].apply(
            lambda x: x
            if x >= 0
            else self.df_performances.loc[self.winning_model]["model_config"].get(
                "floor", 0
            )
        )

        return pred

    def create_all_models(self, models_config: dict):
        """
        # TODO: Doc
        :return:
        """

        # For each model config create its instance
        models = {}
        for model_name, model_config in models_config.items():
            model_flavor = model_name.split("_")[0]
            model = self.model_generation(
                model_flavor=model_flavor, model_config=model_config
            )
            # Add model to the model list
            models[model_name] = model
        return models

    def model_generation(self, model_flavor: str, model_config: dict):
        """
        # TODO: DOC
        :param model_flavor:
        :param model_config:
        :return:
        """
        if model_flavor == "prophet":
            model = KRNSProphet(
                train_data=self.train_data,
                test_data=self.test_data,
                key_column=self.key_col,
                date_col=self.date_col,
                metric_col=self.metric_col,
                interval_width=model_config["interval_width"],
                growth=model_config["growth"],
                daily_seasonality=model_config["daily_seasonality"],
                weekly_seasonality=model_config["weekly_seasonality"],
                yearly_seasonality=model_config["yearly_seasonality"],
                seasonality_mode=model_config["seasonality_mode"],
                floor=model_config["floor"],
                cap=self.max_value,
                country_holidays=model_config["country_holidays"],
            )
        elif model_flavor == "pmdarima":
            model = KRNSPmdarima(
                train_data=self.train_data,
                test_data=self.test_data,
                key_column=self.key_col,
                date_col=self.date_col,
                metric_col=self.metric_col,
                m=model_config["m"],
                seasonal=model_config["seasonal"],
            )
        else:
            raise ValueError(f"Model {model_flavor} not supported.")

        return model
