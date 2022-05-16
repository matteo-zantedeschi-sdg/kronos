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
import json

logger = logging.getLogger(__name__)


def forecast_udf_gen(client: MlflowClient,
                     key_col: str,
                     date_col: str,
                     metric_col: str,
                     quality_col: str,
                     action_col: str,
                     models_col: str,
                     models_config: str,
                     days_from_last_obs_col: str,
                     current_date: datetime.date,
                     fcst_first_date: datetime.date,
                     n_test: int,
                     n_unit_test: int,
                     forecast_horizon: int,
                     dt_creation_col: str,
                     dt_reference_col: str
                     ):
    """
    A function used to create a pandas User Defined Function (UDF) with the specified parameters.

    :param MlflowClient client: The mlflow client to use to track experiments and register models.
    :param str key_col: The name of the column indicating the time series key.
    :param str date_col: The name of the column indicating the time dimension.
    :param str metric_col: The name of the column indicating the dimension to forecast.
    :param str quality_col: The name of the column indicating the quality of time series ("bad" or "good") which determines if a statistical or naive model should be trained.
    :param str action_col: The name of the column indicating the action to take for the time series. This could be:
        * "competition": One (or more) models are trained and put in competition with each other and with the one already in production.
        * "training": Retraining the currently production model.
        * "predict": Use the production model to predict.
    :param str models_col: The name of the column indicating the models for competition.
    :param str models_config: The string (in json format) containing all the model configurations.
    :param str days_from_last_obs_col: The name of the column indicating the days since the last available observation in the time series.
    :param datetime.date current_date: Current processing date.
    :param datetime.date fcst_first_date: Date of first day of forecast, usually is the day following the current date.
    :param int n_test: Number of observations to use as test set.
    :param int n_unit_test: Number of points to forecast in the model deployment's unit test.
    :param int forecast_horizon: Number of points to forecast.
    :param str dt_creation_col: The name of the column indicating the forecast creation date.
    :param str dt_reference_col: The name of the column indicating the date used as reference date for forecast.

    :return: A pandas UDF with the specified parameters as arguments.

    **Example**

    .. code-block:: python

        forecast_udf = forecast_udf_gen(
                client=MlflowClient(),
                key_col='id',
                date_col='date',
                metric_col='y',
                quality_col='quality',
                action_col='action',
                models_col='models',
                models_config='models_config',
                days_from_last_obs_col='days_from_last_obs',
                current_date=(date.today() + timedelta(-1)).strftime('%Y-%m-%d'),
                fcst_first_date=date.today().strftime('%Y-%m-%d'),
                n_test=7,
                n_unit_test=7,
                forecast_horizon=7,
                dt_creation_col='creation_date',
                dt_reference_col='reference_date'
            )

    """
    # Define Pandas UDF
    def forecast_udf(data: pd.DataFrame) -> pd.DataFrame:
        """
        A pandas User Defined Function (UDF) which provides the forecast of a time series and handles the
        interaction with mlflow to track experiments and version the models.

        :param pd.DataFrame data: The pandas DataFrame containing all the information.

        :return: (pd.DataFrame) The pandas DataFrame with forecasted values.

        **Example**

        .. code-block:: python

            partition_key = 'id'
            partition_number = df.select(partition_key).distinct().count()
            df = df.repartition(partition_number, partition_key)

            df_pred = df.groupby(partition_key).applyInPandas(forecast_udf, schema=result_schema)

        """

        # Define all parameters
        future_only = True

        # Forecast parameters
        _key_col = key_col
        _date_col = date_col
        _metric_col = metric_col
        _quality_col = quality_col
        _action_col = action_col
        _models_col = models_col
        _models_config = json.loads(models_config)
        _fcst_col = 'volume_giorno_fcst'
        _days_from_last_obs_col = days_from_last_obs_col
        _current_date = current_date
        _fcst_first_date = fcst_first_date
        _n_test = n_test
        _n_unit_test = n_unit_test
        _forecast_horizon = forecast_horizon
        _dt_creation_col = dt_creation_col
        _dt_reference_col = dt_reference_col

        # Get statistics
        min_value = data[metric_col].min()
        max_value = data[metric_col].max()

        # Set model parameters
        # TODO: Sarà da estendere a più di un modello
        model_flavor = _models_config['prophet_1']['model_flavor']
        interval_width = _models_config['prophet_1']['interval_width']
        growth = _models_config['prophet_1']['growth']
        daily_seasonality = _models_config['prophet_1']['daily_seasonality']
        weekly_seasonality = _models_config['prophet_1']['weekly_seasonality']
        yearly_seasonality = _models_config['prophet_1']['yearly_seasonality']
        seasonality_mode = _models_config['prophet_1']['seasonality_mode']
        floor = _models_config['prophet_1']['floor']
        cap = max_value * 10
        country_holidays = _models_config['prophet_1']['country_holidays']

        # Retrieve key (to later add to the output)
        key_code = str(data[_key_col].iloc[0])
        logger.error(f"####### Working on pdr {key_code}")

        # Retrieve training/predictions parameters
        action = data[_action_col].iloc[0]
        quality = data[_quality_col].iloc[0]

        # Training #####
        if quality == 'good' and action in ['competition', 'training']:
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
                        seasonality_mode=seasonality_mode,
                        floor=floor,
                        cap=cap,
                        country_holidays=country_holidays
                    )

                    # Preprocess
                    krns_prophet.preprocess()

                    # Log model params
                    for key, val in krns_prophet.model_params.items():
                        client.log_param(run_id, key, val)

                    # Fit the model
                    krns_prophet.fit()

                    # Get the model signature and log the model
                    # signature = infer_signature(train_data, krns_prophet.predict(n_days=n_test))
                    # TODO: Signature da aggiungere in futuro, e capire quale
                    mlflow.prophet.log_model(pr_model=krns_prophet.model, artifact_path='model')

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
                if action == 'competition':
                    try:
                        # Retrieve the model
                        prod_model = mlflow.prophet.load_model(f"models:/{key_code}/Production")

                        # Predict with current production model (on test set)
                        last_prod_model_date = prod_model.history_dates[0].date()
                        # TODO: Da rendere agnostico da prophet
                        last_test_date = test_data.sort_values(by='ds', ascending=False, inplace=False).iloc[0]['ds']
                        difference = (last_test_date - last_prod_model_date).days
                        pred_config = prod_model.make_future_dataframe(periods=difference, freq='d', include_history=False)

                        # Add floor and cap
                        pred_config['floor'] = floor
                        pred_config['cap'] = cap

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
                    unit_test_status = MLFlower.unit_test_model(model_version=model_version, n=_n_unit_test, floor=floor,
                                                                cap=cap)

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
                    (datetime.date.today() + datetime.timedelta(days=_forecast_horizon)) - last_date).days
            pred_config = model.make_future_dataframe(periods=actual_forecast_horizon, freq='d', include_history=False)

            # Add floor and cap
            pred_config['floor'] = floor
            pred_config['cap'] = cap

            pred = model.predict(pred_config)
            pred = pred[['ds', 'yhat']]

            # Rename from prophet naming
            pred.rename(columns={"ds": _date_col, "yhat": _fcst_col}, inplace=True)

            # Convert to date
            pred[_date_col] = pred[_date_col].dt.date

        except Exception as e:
            logger.error(f"Prediction failed: {e}")

            # Get last value and date
            last_value = data.sort_values(by=_date_col, ascending=False, inplace=False).iloc[0][_metric_col]
            last_date = data.sort_values(by=_date_col, ascending=False, inplace=False).iloc[0][_date_col]

            # Create dummy pred df
            actual_forecast_horizon = (
                    (datetime.date.today() + datetime.timedelta(days=_forecast_horizon)) - last_date).days
            pred = pd.DataFrame({
                _date_col: [last_date + datetime.timedelta(days=x) for x in range(actual_forecast_horizon)],
                _fcst_col: [last_value for i in range(actual_forecast_horizon)]
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

        # Keep only future rows
        if future_only:
            pred = pred[pred[_date_col] >= _fcst_first_date]

        # Flat predictions to floor value
        # TODO: Da rivedere/discutere
        pred[_fcst_col] = pred[_fcst_col].apply(lambda x: x if x >= 0 else floor)

        return pred

    return forecast_udf
