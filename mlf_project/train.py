from prophet import Prophet
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from modeler import Modeler
from ml_flower import MLFlower
import click
import mlflow
import mlflow.prophet
import pandas as pd
import os
import logging
import time
import datetime

logger = logging.getLogger(__name__)


@click.command(help="Train and prediction of a model.")
@click.argument("data")
@click.argument("run_id")
def run(data: str = "", run_id: str = "") -> None:

    # Define all parameters
    # TODO: Questi vanno tutti letti da qualche parte (control plane, etc).
    #  Uno alla volta li dobbiamo togliere da qui
    date_col = 'date'
    key_col = 'pdr'
    metric_col = 'volume_giorno'
    n_test = 7
    unit_test_days = 7
    forecast_horizon = 7
    model_flavor = 'prophet'
    interval_width = 0.95
    growth = 'linear'
    daily_seasonality = False
    weekly_seasonality = True
    yearly_seasonality = True
    seasonality_mode = 'multiplicative'

    # Init mlflow client
    client = MlflowClient()

    # Convert received data in a Pandas DataFrame
    data = pd.read_json(data)

    # Retrieve key (to later add to the output)
    key_code = data[key_col].iloc[0]

    # Prophet specific pre-processing
    data.rename(columns={date_col: "ds", metric_col: "y"}, inplace=True)

    # Training #####
    try:
        # Train/Test split
        train_data, test_data = Modeler.train_test_split(data=data, date_column="ds", n_test=n_test)

        # Define prophet model
        model = Prophet(interval_width=0.95, daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True,
                        seasonality_mode='multiplicative')

        # Log model params: in general, log all params different from default
        mlflow.log_param("interval_width", interval_width)
        mlflow.log_param("daily_seasonality", daily_seasonality)
        mlflow.log_param("weekly_seasonality", weekly_seasonality)
        mlflow.log_param("yearly_seasonality", yearly_seasonality)
        mlflow.log_param("seasonality_mode", seasonality_mode)

        # Fit the model
        model.fit(train_data)

        # Get the model signature and log the model
        signature = infer_signature(train_data, model.predict(train_data))
        mlflow.prophet.log_model(pr_model=model, artifact_path='model', signature=signature)

        # Make predictions
        pred_config = model.make_future_dataframe(periods=n_test, freq='d', include_history=False)
        pred = model.predict(pred_config)

        # Compute rmse
        # TODO: Andranno calcolate anche eventuali metriche aggiuntive
        test_pred_vs_actual = pd.merge(test_data, pred[['ds', 'yhat']], how='inner', on='ds')
        train_rmse = Modeler.evaluate_model(data=test_pred_vs_actual, metric='rmse', pred_col='yhat', actual_col='y')
        mlflow.log_metric("rmse", train_rmse)

        # Check if a production model already exist and it is still the best one
        prod_model_win = False
        try:
            # Retrieve the model
            prod_model = mlflow.prophet.load_model(f"models:/{key_code}/Production")

            # Predict with current production model (on test set)
            last_prod_model_date = prod_model.history_dates[0].date()
            last_test_date = test_data.sort_values(by=date_col, ascending=False, inplace=False).iloc[0][date_col]
            difference = (last_test_date - last_prod_model_date).days
            pred_config = prod_model.make_future_dataframe(periods=difference, freq='d', include_history=False)
            pred = prod_model.predict(pred_config)

            # Compute rmse
            # TODO: Andranno calcolate anche eventuali metriche aggiuntive
            prod_pred_vs_actual = pd.merge(test_data, pred[['ds', 'yhat']], how='inner', on='ds')
            prod_rmse = Modeler.evaluate_model(data=prod_pred_vs_actual, metric='rmse', pred_col='yhat', actual_col='y')

            # Compute final score and compare
            # TODO: Dovrà essere la somma di tutte le metriche con cui si vogliono confrontare i modelli
            train_score = -train_rmse
            prod_score = -prod_rmse

            # Compare score
            if prod_score > train_score:
                prod_model_win = True
                logger.info("Prod model is still winning.")

        except Exception as e:
            logger.warning(e)

        if not prod_model_win:
            # Register the trained model to MLflow Registry
            model_details = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=str(key_code))
            # Check registration Status
            for _ in range(10):
                model_version_details = client.get_model_version(name=model_details.name, version=model_details.version)
                status = ModelVersionStatus.from_string(model_version_details.status)
                if status == ModelVersionStatus.READY:
                    break
                time.sleep(1)

            # Set the flavor tag
            client.set_model_version_tag(name=model_version_details.name, version=model_version_details.version,
                                         key='model_flavor', value=model_flavor)

            # Transition to "staging" stage and archive the last one (if present)
            model_version = client.transition_model_version_stage(
                name=model_version_details.name,
                version=model_version_details.version,
                stage='staging',
                archive_existing_versions=True
            )

            # Unit test the model
            unit_test_status = MLFlower.unit_test_model(model_version=model_version, unit_test_days=unit_test_days)

            # Deploy model in Production
            if unit_test_status == 'OK':
                deploy_status = MLFlower.deploy_model(client=client, model_version=model_version)

    except Exception as e:
        logger.error(f"Training failed: {e}")

    # Prediction #####
    try:
        # Retrieve production model
        model = mlflow.prophet.load_model(f"models:/{key_code}/Production")

        # Predict with current production model: compute actual forecast horizon needed first
        last_date = model.history_dates[0].date()
        actual_forecast_horizon = (
                (datetime.date.today() + datetime.timedelta(days=forecast_horizon)) -
                (last_date - datetime.timedelta(days=n_test))
        ).days
        pred_config = model.make_future_dataframe(periods=actual_forecast_horizon, freq='d', include_history=False)
        pred = model.predict(pred_config)
        pred = pred[['ds', 'yhat']]

    except Exception as e:
        logger.error(f"Prediction failed: {e}")

        # Get last value and date
        last_value = data.sort_values(by=date_col, ascending=False, inplace=False).iloc[0][metric_col]
        last_date = data.sort_values(by=date_col, ascending=False, inplace=False).iloc[0][date_col]
        # Create dummy pred df
        actual_forecast_horizon = ((datetime.date.today() + datetime.timedelta(days=forecast_horizon)) - last_date).days
        pred = pd.DataFrame({
            'ds': [last_date + datetime.timedelta(days=x) for x in range(actual_forecast_horizon)],
            'yhat': [last_value for i in range(actual_forecast_horizon)]
        })

    # Add key code to predictions
    pred[key_col] = key_code

    # Write results #####
    try:
        # Write predictions to local csv
        file_name = f"/tmp/{key_code}.csv"
        pred.to_csv(file_name, sep=";", index=False)
        if os.path.isfile(file_name):
            logger.debug("File correctly written.")

        # Download azcopy
        az_path = "/tmp/azcopy"
        is_fil = os.path.isfile(az_path)
        logger.debug(f"File already exists? {str(is_fil)}")
        if not is_fil:
            try:
                os.system('wget -O azcopy_v10.tar.gz https://aka.ms/downloadazcopy-v10-linux && tar -xf azcopy_v10.tar.gz -C /tmp/ --strip-components=1')
            except Exception as e:
                logger.error(f"azcopy download failed: {e}")

        # Move file to Azure Storage account
        # TODO: The token should be a secret
        # TODO: The storage path should be a parameter
        storage_path = "https://dlssirisweuts001.blob.core.windows.net/dpc-datascience/dpc_pdr_forecast_temp_landing/"
        token = "?sp=racwdlm&st=2022-04-05T17:15:50Z&se=2022-07-01T01:15:50Z&spr=https&sv=2020-08-04&sr=c&sig=X%2B5Zsp8pGn9RCi9NEsV%2BGmLh%2BAvjEnduVfCYI%2Bj1bAg%3D"
        storage_token = storage_path + token
        os.system(f'{az_path} copy "{file_name}" "{storage_token}" --from-to LocalBlob')
        logger.debug("Results written!")

    except Exception as e:
        logger.error(f"Write results failed: {e}")


if __name__ == "__main__":
    run()
