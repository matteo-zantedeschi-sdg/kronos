from kronos.ml_flower import MLFlower
from kronos.modeler import Modeler
from mlflow.tracking import MlflowClient
import pandas as pd
import datetime
import logging
import json

logger = logging.getLogger(__name__)


def forecast_udf_gen(
    client: MlflowClient,
    key_col: str,
    date_col: str,
    metric_col: str,
    fcst_col: str,
    quality_col: str,
    action_col: str,
    models_col: str,
    models_config: str,
    days_from_last_obs_col: str,
    current_date: datetime.date,
    fcst_first_date: datetime.date,
    n_test: int,
    n_unit_test: int,
    fcst_horizon: int,
    dt_creation_col: str,
    dt_reference_col: str,
):
    """
    A function used to create a pandas User Defined Function (UDF) with the specified parameters.

    :param MlflowClient client: The mlflow client to use to track experiments and register models.
    :param str key_col: The name of the column indicating the time series key.
    :param str date_col: The name of the column indicating the time dimension.
    :param str metric_col: The name of the column indicating the dimension to forecast.
    :param str fcst_col: The name of the column indication the forecast.
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
    :param int fcst_horizon: Number of points to forecast.
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
                fcst_col='y_hat',
                quality_col='quality',
                action_col='action',
                models_col='models',
                models_config='models_config',
                days_from_last_obs_col='days_from_last_obs',
                current_date=(date.today() + timedelta(-1)).strftime('%Y-%m-%d'),
                fcst_first_date=date.today().strftime('%Y-%m-%d'),
                n_test=7,
                n_unit_test=7,
                fcst_horizon=7,
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

        # Retrieve key (to later add to the output)
        key_code = str(data[key_col].iloc[0])
        logger.info(f"### Working on pdr {key_code}")

        # Retrieve training/predictions parameters
        action = data[action_col].iloc[0]
        quality = data[quality_col].iloc[0]

        # Init an ml_flower instance
        ml_flower = MLFlower(client=client)
        # Init a modeler instance
        modeler = Modeler(
            ml_flower=ml_flower,
            data=data,
            key_col=key_col,
            date_col=date_col,
            metric_col=metric_col,
            fcst_col=fcst_col,
            models_config=json.loads(models_config),
            days_from_last_obs_col=days_from_last_obs_col,
            current_date=current_date,
            fcst_first_date=fcst_first_date,
            n_test=n_test,
            n_unit_test=n_unit_test,
            fcst_horizon=fcst_horizon,
            dt_creation_col=dt_creation_col,
            dt_reference_col=dt_reference_col,
        )

        # TRAINING #####
        if quality == "good" and action in ["competition", "training"]:
            modeler.training()

            prod_model_win = False
            modeler.prod_model_eval()

            if action == "competition":
                modeler.competition()
                if modeler.winning_model_name == "prod_model":
                    prod_model_win = True
                    logger.info("### Prod model is still winning.")

            if not prod_model_win:
                modeler.deploy()

        # PREDICTION #####
        pred = modeler.prediction()

        return pred

    return forecast_udf
