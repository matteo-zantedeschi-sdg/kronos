from kronos.ml_flower import MLFlower
from kronos.models.krns_prophet import KRNSProphet
from mlflow.models.signature import infer_signature
from datetime import datetime
import pandas as pd
import mlflow
import mlflow.prophet


class Modeler:
    def __init__(self):
        print("Modeler initialized")

    @staticmethod
    def train_test_split(
        data: pd.DataFrame, date_column: str, n_test: int
    ) -> pd.DataFrame and pd.DataFrame:
        # TODO: Gestire il caso in cui non ci sono sufficienti record per il test

        train_data = data.sort_values(by=[date_column], ascending=False).iloc[
            n_test:, :
        ]
        test_data = data.sort_values(by=[date_column], ascending=False).iloc[:n_test, :]

        return train_data, test_data

    @staticmethod
    def evaluate_model(
        data: pd.DataFrame, metric: str, predicted_col: str, true_value_col: str
    ) -> int:

        # Transform metric in lower case and remove whitespaces
        metric = metric.lower().replace(" ", "")

        if metric not in ["rmse"]:
            print(f"Requested metric {metric} is not supported.")
            print(f"Available metrics are: rmse")

        out = None
        if metric == "rmse":
            out = ((data[true_value_col] - data[predicted_col]) ** 2).mean() ** 0.5

        return out

    @staticmethod
    def forecast(data: pd.DataFrame) -> pd.DataFrame:

        # Parameters
        n_test = 7
        pdr_code = data.iloc[0].pdr
        model_flavor = "prophet"
        run_name = f"{pdr_code}_{model_flavor}_{str(datetime.now()).replace(' ', '_')}"
        interval_width = 0.95
        growth = "linear"
        daily_seasonality = False
        weekly_seasonality = True
        yearly_seasonality = True
        seasonality_mode = "multiplicative"
        unit_test_days = 7
        forecast_horizon = 7

        # Train/Test split
        print("Train/Test split")
        train_data, test_data = Modeler.train_test_split(
            data=data, date_column="date", n_test=n_test
        )

        # Manage experiment
        print("Manage experiment")
        ml_flower = MLFlower()
        experiment_name = f"/mlflow/experiments/{pdr_code}"
        experiment = ml_flower.get_experiment(experiment_name=experiment_name)

        # Start an MLflow run
        # The "with" keyword ensures the run will be closed even if the execution crashes
        print(f"Starting MLflow run: {run_name}")
        # TODO: Da gestire il caso in cui experiment non è un Experiment (non è ancora stato creato)
        with mlflow.start_run(
            experiment_id=experiment.experiment_id, run_name=run_name
        ) as run:

            # Specify in a tag the model flavor
            mlflow.set_tag(key="model_flavor", value=model_flavor)

            # Log all params (TODO: Capire quanto "costa" loggare questi parametri e se nel caso è evitabile)
            # In general, log all params different from default
            print("Logging parameters")
            mlflow.log_param("interval_width", interval_width)
            mlflow.log_param("growth", growth)
            mlflow.log_param("daily_seasonality", daily_seasonality)
            mlflow.log_param("weekly_seasonality", weekly_seasonality)
            mlflow.log_param("yearly_seasonality", yearly_seasonality)
            mlflow.log_param("seasonality_mode", seasonality_mode)

            # Init prophet model
            krns_prophet = KRNSProphet(
                date_column="date",
                key_column="pdr",
                metric_column="volume_giorno",
                model=None,
            )

            # Train prophet model
            krns_prophet.fit(data=train_data)

            # Log model
            # TODO: Da aggiungere anche tutti gli altri attributi che è possibile loggare
            # TODO: Capire qual è il luogo esatto in cui salvare i modelli
            # TODO: Da sistemare l'infer
            print("Log the model")
            signature = infer_signature(
                train_data,
                krns_prophet.model.predict(
                    train_data.rename(columns={"date": "ds", "volume_giorno": "y"})
                ),
            )
            mlflow.prophet.log_model(
                pr_model=krns_prophet.model, artifact_path="model", signature=signature
            )

            # Predict on test set (both trained and production model)
            trained_model_test_pred = krns_prophet.predict(
                data=test_data, periods=n_test, freq="d", include_history=False
            )

            # Log prediction figure - TODO
            # predict_fig = model.plot(test_pred, xlabel='date', ylabel='consumptions').savefig('prophet.png')
            # mlflow.log_artifact("prophet.png")

            prod_model_fl = False
            try:
                prod_model = mlflow.prophet.load_model(f"models:/{pdr_code}/Production")
                prod_model_fl = True
            except Exception as e:
                print(e)

            if prod_model_fl:
                krns_prod_model = KRNSProphet(
                    date_column="date",
                    key_column="pdr",
                    metric_column="volume_giorno",
                    model=prod_model,
                )
                prod_model_test_pred = krns_prod_model.predict(
                    data=test_data, periods=n_test, freq="d", include_history=False
                )

            # Evaluate model
            print("Evaluate the model and log metrics")
            trained_model_rmse = Modeler.evaluate_model(
                data=trained_model_test_pred,
                metric="rmse",
                predicted_col="yhat",
                true_value_col="y",
            )
            # Log metric for trained model
            mlflow.log_metric("rmse", trained_model_rmse)
            if prod_model_fl:
                prod_model_rmse = Modeler.evaluate_model(
                    data=prod_model_test_pred,
                    metric="rmse",
                    predicted_col="yhat",
                    true_value_col="y",
                )

                # Competition

                # Combine all metrics
                # TODO: Now is just rmse, later could be a combination of all metrics
                trained_model_score = -trained_model_rmse
                prod_model_score = -prod_model_rmse

            # Compare score
            if prod_model_fl and prod_model_score > trained_model_score:
                print("Prod model is still winning.")
            else:
                print("New trained model is better than previous production model.")

                # Register model
                model_version = ml_flower.register_model(
                    model_uri=f"runs:/{run.info.run_uuid}/model",
                    model_name=pdr_code,
                    timeout_s=10,
                    model_flavor_tag=model_flavor,
                )

                # Unit test the model
                unit_test_status = ml_flower.unit_test_model(
                    model_version=model_version, unit_test_days=unit_test_days
                )

                # Deploy model
                if unit_test_status == "OK":
                    deploy_status = ml_flower.deploy_model(model_version=model_version)

            # Retrieve production model
            model = mlflow.prophet.load_model(f"models:/{pdr_code}/Production")
            krns_model = KRNSProphet(
                date_column="date",
                key_column="pdr",
                metric_column="volume_giorno",
                model=model,
            )

            # Predict with new model
            pred = krns_model.predict(
                data=data,
                periods=forecast_horizon,
                freq="d",
                include_history=False,
                final_prediction=True,
            )

            return pred
