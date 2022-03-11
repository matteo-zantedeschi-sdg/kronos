"""
Time-Series forecasting using MLflow.
The run method will train a time-series model in a new MLflow run.
The runs are evaluated based on validation set loss. Test set score is calculated to verify the
results.
Several runs can be run in parallel.
"""
import click
import time
import mlflow
import mlflow.prophet
import mlflow.projects
import pandas as pd
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
def run(data_path):
    # Read delta table from path provided
    df = spark.read.format("delta").load(f"{data_path}")

    # Define output schema
    # TODO: to complete
    result_schema = StructType([
        StructField('pdr', StringType())
    ])

    # Define pandas UDF
    def forecast(data: pd.DataFrame) -> pd.DataFrame:

        # Retrieve pdr code
        pdr_code = data['pdr'].iloc[0]

        # Define experiment name
        experiment_name = f'/mlflow/experiments/{pdr_code}'

        # with mlflow.start_run(run_name=pdr_code, nested=True) as child_run:
        try:
            experiment = client.create_experiment(experiment_name)
        except:
            experiment = client.get_experiment_by_name(experiment_name).experiment_id

        p = mlflow.projects.run(
            uri=".",
            entry_point="train",
            parameters={
                "training_data": data.to_json(),
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
    df.groupBy('pdr').applyInPandas(forecast, result_schema).count()

    # for i in range(1, 30):
    #     active_runs = mlflow.list_run_infos(
    #         experiment_id=experiment_id,
    #         run_view_type=ViewType.ACTIVE_ONLY,
    #         max_results=1
    #     )
    #     if len(active_runs) > 0:
    #         time.sleep(10)
    #     else:
    #         break

    # TODO: Get all active runs and wait till their ends


if __name__ == "__main__":
    run()
