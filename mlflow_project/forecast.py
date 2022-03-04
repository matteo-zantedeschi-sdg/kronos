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
import mlflow.tracking
import mlflow.projects
import pandas as pd
from pyspark.sql.types import StructType, StructField, StringType, DateType, FloatType


@click.command(help="Perform forecast for each pdr")
@click.argument("data_path")
def run(data_path):

    # Read data
    df = spark.read.format("delta").load(f"{data_path}")

    # Define output schema
    result_schema = StructType([])
    tracking_client = mlflow.tracking.MlflowClient()

    def forecast(data: pd.DataFrame) -> pd.DataFrame:
        with mlflow.start_run(nested=True) as child_run:
            p = mlflow.projects.run(
                run_id=child_run.info.run_id,
                uri=".",
                entry_point="train",
                parameters={
                    "training_data": data,
                },
                experiment_id=experiment_id,
                backend="databricks",
                use_conda=False,
                synchronous=False,
            )
            succeeded = p.wait()
        if succeeded:
            # training_run = tracking_client.get_run(p.run_id)
            print("Successfully")
        else:
            tracking_client.set_terminated(p.run_id, "FAILED")

        return pd.DataFrame()

    with mlflow.start_run() as run:
        experiment_id = run.info.experiment_id

        df.groupBy('pdr').applyInPandas(forecast, result_schema)


if __name__ == "__main__":
    run()
