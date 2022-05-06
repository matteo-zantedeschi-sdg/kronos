from kronos.ml_flower import MLFlower
from kronos.modeler import Modeler
from kronos.models.krns_prophet import KRNSProphet
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
import mlflow
import mlflow.prophet
import pandas as pd
import time
import datetime
import logging

logger = logging.getLogger(__name__)


def forecast_udf_gen(client: MlflowClient,
                     key_col: str,
                     date_col: str,
                     metric_col: str,
                     days_from_last_obs_col: str,
                     current_date: datetime.date,
                     fcst_first_date: datetime.date,
                     n_test: int,
                     n_unit_test: int,
                     forecast_horizon: int,
                     dt_creation_col: str,
                     dt_reference_col: str
                     ):

    # Define Pandas UDF
    def forecast_udf(data: pd.DataFrame) -> pd.DataFrame:

        # Define all parameters
        # TODO: Questi vanno tutti letti da qualche parte (control plane, etc).
        #  Uno alla volta li dobbiamo togliere da qui
        future_only = True

        # Forecast parameters
        _key_col = key_col
        _date_col = date_col
        _metric_col = metric_col
        _days_from_last_obs_col = days_from_last_obs_col
        _current_date = current_date
        _fcst_first_date = fcst_first_date
        _n_test = n_test
        _n_unit_test = n_unit_test
        _forecast_horizon = forecast_horizon
        _dt_creation_col = dt_creation_col
        _dt_reference_col = dt_reference_col

        # Model parameters
        # TODO: Servirà passare il json
        model_flavor = 'prophet'
        interval_width = 0.95
        growth = 'linear'
        daily_seasonality = False
        weekly_seasonality = True
        yearly_seasonality = True
        seasonality_mode = 'multiplicative'

        # Retrieve key (to later add to the output)
        key_code = str(data[_key_col].iloc[0])

        # Training #####
        try:
            # Define experiment path
            experiment_path = f'/mlflow/experiments/{key_code}'
            # Create/Get experiment
            try:
                experiment = client.create_experiment(experiment_path)
            except Exception as e:
                print(e)
                experiment = client.get_experiment_by_name(experiment_path).experiment_id

            # Start run
            run_name = datetime.datetime.utcnow().isoformat()
            with mlflow.start_run(experiment_id=experiment, run_name=run_name) as run:

                # Store run id
                run_id = run.info.run_id

                # Train/Test split
                train_data, test_data = Modeler.train_test_split(data=data, date_col=_date_col, n_test=_n_test)

                # Init kronos prophet
                krns_prophet = KRNSProphet(
                    train_data=train_data,
                    test_data=test_data,
                    key_column=_key_col,
                    date_col=_date_col,
                    metric_col=_metric_col,
                    interval_width=interval_width,
                    growth=growth,
                    daily_seasonality=daily_seasonality,
                    weekly_seasonality=weekly_seasonality,
                    yearly_seasonality=yearly_seasonality,
                    seasonality_mode=seasonality_mode
                )

                # Preprocess
                krns_prophet.preprocess()

                # Log model params
                for key, val in krns_prophet.model_params.items():
                    client.log_param(run_id, key, val)

                # Fit the model
                krns_prophet.fit()

                # Get the model signature and log the model
                signature = infer_signature(train_data, krns_prophet.model.predict(train_data))
                mlflow.prophet.log_model(pr_model=krns_prophet.model, artifact_path='model', signature=signature)

                # Make predictions
                pred = krns_prophet.predict(n_days=_n_test)

                # Compute rmse
                # TODO: Andranno calcolate anche eventuali metriche aggiuntive - lette da una tabella parametrica
                # TODO: yhat è tipica di prophet, da generalizzare
                train_rmse = Modeler.evaluate_model(actual=test_data, pred=pred, metric='rmse', pred_col='yhat',
                                                    actual_col='y')
                client.log_metric(run_id, "rmse", train_rmse)

            # Check if a production model already exist and it is still the best one
            prod_model_win = False
            try:
                # Retrieve the model
                prod_model = mlflow.prophet.load_model(f"models:/{key_code}/Production")

                # Predict with current production model (on test set)
                last_prod_model_date = prod_model.history_dates[0].date()
                # TODO: Da rendere agnostico da prophet
                last_test_date = test_data.sort_values(by='ds', ascending=False, inplace=False).iloc[0]['ds']
                difference = (last_test_date - last_prod_model_date).days
                pred_config = prod_model.make_future_dataframe(periods=difference, freq='d', include_history=False)
                pred = prod_model.predict(pred_config)

                # Compute rmse
                # TODO: Andranno calcolate anche eventuali metriche aggiuntive
                prod_rmse = Modeler.evaluate_model(actual=test_data, pred=pred, metric='rmse', pred_col='yhat',
                                                   actual_col='y')

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
                model_details = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=key_code)
                # Check registration Status
                for _ in range(10):
                    model_version_details = client.get_model_version(name=model_details.name,
                                                                     version=model_details.version)
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
                unit_test_status = MLFlower.unit_test_model(model_version=model_version, n=_n_unit_test)

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
                    (datetime.date.today() + datetime.timedelta(days=forecast_horizon)) - last_date).days
            pred_config = model.make_future_dataframe(periods=actual_forecast_horizon, freq='d', include_history=False)
            pred = model.predict(pred_config)
            pred = pred[['ds', 'yhat']]

        except Exception as e:
            logger.error(f"Prediction failed: {e}")

            # Get last value and date
            last_value = data.sort_values(by=_date_col, ascending=False, inplace=False).iloc[0][_metric_col]
            last_date = data.sort_values(by=_date_col, ascending=False, inplace=False).iloc[0][_date_col]
            # Create dummy pred df
            actual_forecast_horizon = (
                    (datetime.date.today() + datetime.timedelta(days=forecast_horizon)) - last_date).days
            pred = pd.DataFrame({
                'ds': [last_date + datetime.timedelta(days=x) for x in range(actual_forecast_horizon)],
                'yhat': [last_value for i in range(actual_forecast_horizon)]
            })

        # Compute days from last obs, reference date and prediction date
        last_date = data.sort_values(by=_date_col, ascending=False, inplace=False).iloc[0][_date_col]
        days_from_last_obs = (datetime.date.today() - last_date).days
        # Add to predictions
        pred[days_from_last_obs_col] = days_from_last_obs
        pred[_dt_reference_col] = _fcst_first_date
        pred[_dt_creation_col] = _current_date

        # Add key code to predictions
        pred[key_col] = key_code

        # Rename from prophet naming
        pred.rename(columns={"ds": _date_col, "yhat": "volume_giorno_fcst"}, inplace=True)

        # Convert to date
        pred[_date_col] = pred[_date_col].dt.date

        # Keep only future rows
        if future_only:
            pred = pred[pred[_date_col] >= _fcst_first_date]

        return pred

    return forecast_udf
