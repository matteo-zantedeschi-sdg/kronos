from prophet import Prophet
from mlflow.tracking import MlflowClient
import click
import mlflow
import mlflow.prophet
import pandas as pd


@click.command(help="Train and prediction of a model.")
@click.argument("data")
@click.argument("run_id")
def run(data: str = "", run_id: str = "") -> None:

    # Init mlflow client
    client = MlflowClient()

    # Convert received data in a Pandas DataFrame
    data = pd.read_json(data)

    # Retrieve key (to later add to the output)
    key = data['key'].iloc[0]

    # Define prophet model
    model = Prophet(interval_width=0.95, growth='linear', daily_seasonality=False, weekly_seasonality=True,
                    yearly_seasonality=True, seasonality_mode='multiplicative')
    # Fit the model
    model.fit(data)

    # Log a param to mlflow
    client.log_param(run_id, "key", str(key))
    # Log the model to mlflow
    mlflow.prophet.log_model(pr_model=model, artifact_path='model')
    # Log a metric
    client.log_metric(run_id, "rmse", 1.1)

    # Make predictions
    pred_conf = model.make_future_dataframe(periods=7, freq='d', include_history=False)
    pred = model.predict(pred_conf)
    pred = pred[['ds', 'yhat']]
    pred['key'] = key


if __name__ == "__main__":
    run()
