"""
Time-Series forecasting using MLflow.
The run method will train a time-series model in a new MLflow run.
The runs are evaluated based on validation set loss. Test set score is calculated to verify the
results.
Several runs can be run in parallel.
"""
import click
import mlflow
import mlflow.prophet
import mlflow.projects
import pandas as pd
import time
from mlflow.tracking import MlflowClient
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from mlflow.entities import ViewType
from pyspark.sql.types import StructType, StructField, StringType, DateType, FloatType
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

sc = SparkContext('local')
spark = SparkSession(sc)


@click.command(help="Distribute forecasting through a pandas UDF")
@click.argument("data_path")
@click.argument("experiment_name")
def run(data_path: str, experiment_name: str):

    # Read delta table from path provided
    df = spark.read.format("delta").load(f"{data_path}")

    # Define output schema
    # TODO: to complete
    result_schema = StructType([
        StructField('pdr', StringType())
    ])

    # Define pandas UDF
    def forecast(data: pd.DataFrame) -> pd.DataFrame:

        # Create child run
        child_run = client.create_run(
            experiment_id=experiment,
            tags={
                MLFLOW_PARENT_RUN_ID: parent_run_id
            },
        )

        # Launch train
        p = mlflow.projects.run(
            run_id=child_run.info.run_id,
            uri=".",
            entry_point="train",
            parameters={
                "data": data.to_json(),
                "run_id": child_run.info.run_id
            },
            experiment_id=experiment,
            backend="local",
            use_conda=False,
            synchronous=False,
        )

        # Just a placeholder to use pandas UDF and distribute work
        out = pd.DataFrame(data={'pdr': ['1']})

        return out

    # Define the client
    client = MlflowClient()

    # Define experiment name
    experiment_path = f'/mlflow/experiments/{experiment_name}'

    # Create/Get experiment
    try:
        experiment = client.create_experiment(experiment_path)
    except:
        experiment = client.get_experiment_by_name(experiment_path).experiment_id

    # Create parent run
    parent_run = client.create_run(experiment_id=experiment)
    parent_run_id = parent_run.info.run_id

    # Launch pandas UDF
    df.groupBy('pdr').applyInPandas(forecast, result_schema).count()

    # Wait until all runs finish
    # TODO: Da gestire un timeout massimo
    for i in range(1, 30):
        active_runs = mlflow.list_run_infos(
            experiment_id=experiment,
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=2
        )
        if len(active_runs) > 1:
            print("still active")
            time.sleep(10)
        else:
            break


if __name__ == "__main__":
    run()
