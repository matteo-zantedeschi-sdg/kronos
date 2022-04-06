from prophet import Prophet
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.entities.model_registry import ModelVersion
# from modeler import Modeler
# from ml_flower import MLFlower
# from pyspark.context import SparkContext
# from pyspark.sql.session import SparkSession
# from pyspark.sql import SparkSession
import click
import mlflow
import mlflow.prophet
# from mlflow import prophet
import pandas as pd
import copy
import time
import logging
import datetime
import os


# TODO: Da inserire gestione eccezioni
# TODO: Da internalizzare quanto possibile in Modeler e MLFlower
# TODO: Aggiungere la documentazione
# TODO: Da definire il log
# TODO: Inseriamo un modello dummy (può essere anche una ARIMA semplice o ets) per i PDR che hanno troppi pochi dati
#  per un modello più complesso (check data quality)
# TODO: Andrà letta la tabella pdr_control_plane
# TODO: La tabella di forecast avrà anche una colonna che mi indica (per ogni giorno) la distanza della data di quella
#  previsione dall'ultima osservazione disponibile per quel pdr.
# TODO: Inserire in modo parametrico una data di riferimento (dt_riferimento) a cui "ancorarsi" per
#  la previsione: orizzonte temporale, etc
# TODO: Schema tabella output:
#  pdr, societa, arera, giorno_gas, volume_giorno_fcst, dt_creazione, dt_riferimento, giorni_da_ultima_osservazione

@click.command(help="Train/Prediction of a model.")
@click.argument("data")
@click.argument("run_id")
@click.argument("delta_output_path")
@click.argument("train_fl")
@click.argument("prediction_fl")
def run(data: str = "",
        run_id: str = "",
        delta_output_path: str = "",
        train_fl: bool = True,
        prediction_fl: bool = True
        ) -> None:

    # Define spark context
    # sc = SparkContext('local')
    # spark = SparkSession(sc)
    # spark = SparkSession.builder.config("spark.port.maxRetries", "200").getOrCreate()
    # spark = SparkSession.builder.appName('forecast-test').getOrCreate()
    # spark = SparkSession.builder.appName('forecast-test').getOrCreate()
    print("CIAO")

    client = MlflowClient()

    # Suppress warnings (specific to prophet)
    # logger = spark._jvm.org.apache.log4j
    # logging.getLogger("py4j.java_gateway").setLevel(logging.ERROR)

    def train_test_split(data: pd.DataFrame, date_column: str, n_test: int) -> pd.DataFrame and pd.DataFrame:
        # TODO: Gestire il caso in cui non ci sono sufficienti record per il test

        train_data = data.sort_values(by=[date_column], ascending=False).iloc[n_test:, :]
        test_data = data.sort_values(by=[date_column], ascending=False).iloc[:n_test, :]

        return train_data, test_data

    def evaluate_model(data: pd.DataFrame, metric: str, predicted_col: str, true_value_col: str) -> int:

        # Transform metric in lower case and remove whitespaces
        metric = metric.lower().replace(" ", "")

        if metric not in ['rmse']:
            print(f"Requested metric {metric} is not supported.")
            print(f"Available metrics are: rmse")

        out = None
        if metric == 'rmse':
            out = ((data[true_value_col] - data[predicted_col]) ** 2).mean() ** .5

        return out

    def unit_test_model(model_version: ModelVersion, unit_test_days: int) -> str:
        # Load the current staging model and test its prediction method

        # Retrieve model flavor tag
        # model_flavor_tag = model_version.tags['model_flavor']

        # Unit test
        # if model_flavor_tag == 'prophet':
        # Retrieve model and make predictions
        model = mlflow.prophet.load_model(f"models:/{model_version.name}/{model_version.current_stage}")
        pred_conf = model.make_future_dataframe(periods=unit_test_days, freq='d', include_history=False)
        pred = model.predict(pred_conf)
        # else:
        # print(f"Model flavor {model_flavor_tag} not managed.")
        # pred = None

        # Check quality
        out = 'OK' if len(pred) == unit_test_days else 'KO'

        return out

    def deploy_model(model_version: ModelVersion) -> str:
        # Take the current staging model and promote it to production
        # Archive the "already in production" model
        # Return status code

        model_version = client.transition_model_version_stage(
            name=model_version.name,
            version=model_version.version,
            stage='production',
            archive_existing_versions=True
        )

        out = 'OK' if model_version.status == 'READY' else 'KO'

        return out

    if train_fl:
        try:
            # Convert received data in a Pandas DataFrame
            data = pd.read_json(data)

            # Retrieve pdr code and log it
            pdr_code = data['pdr'].iloc[0]
            mlflow.log_param('pdr', pdr_code)

            # Define all parameters
            # TODO: Dovranno essere passati dall'esterno
            n_test = 7
            unit_test_days = 7
            forecast_horizon = 7
            date_col = 'date'
            key_col = 'pdr'
            metric_col = 'volume_giorno'
            model_flavor = 'prophet'
            interval_width = 0.95
            growth = 'linear'
            daily_seasonality = False
            weekly_seasonality = True
            yearly_seasonality = True
            seasonality_mode = 'multiplicative'

            # Prophet specific pre-processing
            renamed_data = copy.deepcopy(data)
            renamed_data = renamed_data.rename(columns={date_col: "ds", metric_col: "y"})

            # Train/Test split
            train_data, test_data = train_test_split(data=renamed_data, date_column="ds", n_test=n_test)

            # Specify model flavor tag
            mlflow.set_tag(key='model_flavor', value=model_flavor)

            # Log all params: in general, log all params different from default
            # TODO: Esplorare auto-log
            mlflow.log_param("interval_width", interval_width)
            mlflow.log_param("growth", growth)
            mlflow.log_param("daily_seasonality", daily_seasonality)
            mlflow.log_param("weekly_seasonality", weekly_seasonality)
            mlflow.log_param("yearly_seasonality", yearly_seasonality)
            mlflow.log_param("seasonality_mode", seasonality_mode)

            # Define the model
            # TODO: Da verificare che non otteniamo previsioni negative, altrimenti dovremo inserire dei cap
            model = Prophet(
                interval_width=0.95,
                growth='linear',
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                seasonality_mode='multiplicative'
            )

            # Fit the model
            model.fit(train_data)

            # Get the model signature and log the model
            signature = infer_signature(train_data, model.predict(train_data))
            mlflow.prophet.log_model(pr_model=model, artifact_path='model', signature=signature)

            # Configure prediction
            pred_conf = model.make_future_dataframe(periods=n_test, freq='d', include_history=False)

            # Make predictions
            pred = model.predict(pred_conf)

            # Combine pred with original test data
            pred = pred[['ds', 'yhat']].set_index('ds')
            test_train_pred = copy.deepcopy(test_data)
            test_train_pred = test_train_pred[['ds', 'y']].set_index('ds')
            test_train_pred = test_train_pred.merge(pred, how='inner', left_index=True, right_index=True).reset_index()

            # Check if a production model already exist
            prod_model_fl = False
            try:
                prod_model = mlflow.prophet.load_model(f"models:/{pdr_code}/Production")
                prod_model_fl = True
            except Exception as e:
                print(e)

            # Predict with current production model
            if prod_model_fl:
                prod_model_pred_conf = prod_model.make_future_dataframe(periods=n_test, freq='d', include_history=False)
                prod_pred = prod_model.predict(prod_model_pred_conf)
                prod_pred = prod_pred[['ds', 'yhat']].set_index('ds')
                test_prod_pred = copy.deepcopy(test_data)[['ds', 'y']].set_index('ds')
                test_prod_pred = test_prod_pred.join(prod_pred, how='left').reset_index(level=0, inplace=True)

            # Evaluate trained model
            train_rmse = evaluate_model(data=test_train_pred,
                                        metric='rmse',
                                        predicted_col='yhat',
                                        true_value_col='y')
            mlflow.log_metric("rmse", train_rmse)

            # Evaluate current production model
            if prod_model_fl:
                prod_rmse = evaluate_model(data=test_prod_pred,
                                           metric='rmse',
                                           predicted_col='yhat',
                                           true_value_col='y')

                # Competition, combining all metrics
                # TODO: Now is just rmse, later could be a combination of all metrics
                trained_model_score = -train_rmse
                prod_model_score = -prod_rmse

            # Compare score
            if prod_model_fl and prod_model_score > trained_model_score:
                print("Prod model is still winning.")
            else:
                print("New trained model is better than previous production model. \n We are going to register it.")

                # Init the MLflow Client
                client = MlflowClient()
                model_details = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=str(pdr_code))

                # Check registration Status
                for _ in range(10):
                    model_version_details = client.get_model_version(name=model_details.name,
                                                                     version=model_details.version)
                    status = ModelVersionStatus.from_string(model_version_details.status)
                    if status == ModelVersionStatus.READY:
                        break
                    time.sleep(1)

                # Set the flavor tag
                client.set_model_version_tag(
                    name=model_version_details.name,
                    version=model_version_details.version,
                    key='model_flavor',
                    value=model_flavor
                )

                # Transition to "staging" stage and archive the last one (if present)
                model_version = client.transition_model_version_stage(
                    name=model_version_details.name,
                    version=model_version_details.version, stage='staging',
                    archive_existing_versions=True
                )

                # Unit test the model
                unit_test_status = unit_test_model(model_version=model_version,
                                                   unit_test_days=unit_test_days)

                # Deploy model
                if unit_test_status == 'OK':
                    deploy_status = deploy_model(model_version=model_version)
                    print(f"Deployment status: {deploy_status}")

        except Exception as e:
            print(e)

    if prediction_fl:

        # Retrieve pdr code
        pdr_code = data['pdr'].iloc[0]

        try:
            # TODO: Se il primo modello non supera lo unit test e non è presente un modello in production?
            # TODO: Oppure se viene lanciato solo in prediction e non c'è un modello in production?

            # Retrieve production model
            model = mlflow.prophet.load_model(f"models:/{pdr_code}/Production")

            # Define forecast configuration
            # TODO: Da cambiare e fare il forecast con orizzonte variabile in base all'ultima osservazione osservata
            pred_conf = model.make_future_dataframe(periods=n_test, freq='d', include_history=False)

            # Prediction
            pred = model.predict(pred_conf)

            # Prepare output to write
            pred = pred[['ds', 'yhat']]

            pred['pdr'] = pdr_code

        except Exception as e:
            print(e)

            # Get last value and date
            last_value = data.sort_values(by=date_col, ascending=False, inplace=False).iloc[0][metric_col]
            last_date = data.sort_values(by=date_col, ascending=False, inplace=False).iloc[0][date_col]

            # Create dummy pred df
            pred_values = [last_value for i in range(7)]
            pred_dates = [last_date - datetime.timedelta(days=x) for x in range(7)]
            pred = pd.DataFrame({'pdr': [pdr_code for i in range(7)], 'ds': pred_dates, 'yhat': pred_values})

        # Convert pandas to Spark Dataframe
        # spark_pred = spark.createDataFrame(pred)

        # Write results in the output delta table
        # spark_pred.write.format("delta").mode("append").save(f"{delta_output_path}")


if __name__ == "__main__":
    run()
