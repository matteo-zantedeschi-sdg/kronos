from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from mlflow.entities import ViewType
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
import click
import mlflow
import mlflow.prophet
import mlflow.projects
import pandas as pd
import time
import re


# TODO: Da definire il log

@click.command(help="Parallel launch of mlflow runs.")
@click.argument("input_path")
@click.argument("output_path")
@click.argument("experiment_name")
@click.argument("train_fl")
@click.argument("prediction_fl")
@click.argument("timeout")
def run(input_path: str = "",
        output_path: str = "",
        experiment_name: str = "",
        train_fl: bool = False,
        prediction_fl: bool = False,
        timeout: int = 10800
        ) -> None:
    """
    Time-Series forecasting using Spark and logging with MLflow.

    The run method, through a pandas UDF, will train in parallel a time-series model for each pdr and will log results
    in a new MLflow run. The several runs are evaluated based on validation set loss.
    If performing, and after a unit test, the trained model is registered in the model registry.
    Finally, the model are retrieved from model registry and used for forecasting.

    :param input_path: The path to the delta table to be read, containing the pdr of which to provide the forecast.
    :param output_path: The path where to write the output delta table, containing the forecasts obtained.
    :param experiment_name: The main experiment name, it should be valued with the execution timestamp in UTC format.
    It will later be used as a suffix of the delta output table.
    :param train_fl: Flag to control if training models.
    :param prediction_fl: Flag to control if predict using models.
    :param timeout: Timeout (in seconds) for the whole execution.
    :return: No return.
    """

    # Define spark context
    sc = SparkContext('local')
    spark = SparkSession(sc)

    # Read delta table from path provided
    df = spark.read.format("delta").load(f"{input_path}")

    # Define output schema (just as a placeholder for the pandas UDF)
    result_schema = StructType([
        StructField('placeholder', StringType())
    ])

    # Define pandas UDF
    def forecast(data: pd.DataFrame) -> pd.DataFrame:

        # TODO: Per "foreachPartition" test
        # data = pd.DataFrame(list(data), columns=("pdr", "date", "volume_giorno"))

        # Create child run
        child_run = client.create_run(experiment_id=experiment,
                                      tags={MLFLOW_PARENT_RUN_ID: parent_run_id}, )

        # Launch train entry
        mlflow.projects.run(
            run_id=child_run.info.run_id,
            uri=".",
            entry_point="train",
            parameters={
                "data": data.to_json(),
                "run_id": child_run.info.run_id,
                "delta_output_path": delta_output_path,
                "train_fl": train_fl,
                "prediction_fl": prediction_fl
            },
            experiment_id=experiment,
            backend="local",
            use_conda=False,
            synchronous=False,
        )

        # Fill the placeholder for pandas UDF
        out = pd.DataFrame(data={'placeholder': ['1']})

        return out

    # Init the MLflow Client
    client = MlflowClient()

    # Define experiment path
    experiment_path = f'/mlflow/experiments/{experiment_name}'

    # Add experiment name as a suffix of the output folder
    parts = re.search('(.*)(/$)', output_path)
    delta_output_path = f"{parts.groups()[0]}_{experiment_name}{parts.groups()[1]}"

    # Create/Get experiment
    try:
        experiment = client.create_experiment(experiment_path)
    except Exception as e:
        print(e)
        experiment = client.get_experiment_by_name(experiment_path).experiment_id

    # Create parent run
    parent_run = client.create_run(experiment_id=experiment)
    parent_run_id = parent_run.info.run_id

    # Launch pandas UDF
    # df.foreachPartition(forecast)
    df.groupBy('pdr').applyInPandas(forecast, result_schema).count()

    # Wait until all runs finish
    # Note: at least one run (the parent) is always running until all children finish
    sleep_s = 10
    for i in range(1, int(timeout / sleep_s)):
        active_runs = mlflow.list_run_infos(
            experiment_id=experiment,
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=2
        )
        if len(active_runs) > 1:
            print("Still active")
            time.sleep(secs=sleep_s)
        else:
            break


if __name__ == "__main__":
    run()
