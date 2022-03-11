"""
Train a simple Prophet model.
"""

import click
import warnings
import mlflow
import mlflow.prophet
import pandas as pd
from prophet import Prophet


@click.command(help="Trains a Prophet model." "The model and its metrics are logged with mlflow.")
@click.argument("training_data")
def run(training_data: str) -> None:
    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Convert received data as Pandas DataFrame
    training_data = pd.read_json(training_data)

    # Read pdr code
    pdr_code = training_data['pdr'].iloc[0]

    # Log pdr as param
    mlflow.log_param('pdr', pdr_code)

    # Prophet specific pre-processing
    training_data = training_data.rename(columns={'date': "ds", 'volume_giorno': "y"})

    # Define the model
    model = Prophet(
        interval_width=0.95,
        growth='linear',
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        seasonality_mode='multiplicative'
    )

    # Fit the model
    model.fit(training_data)

    # Log the model
    mlflow.prophet.log_model(model, "model")

    # configure predictions
    # future_pd = model.make_future_dataframe(
    #     periods=7,
    #     freq='d',
    #     include_history=False
    # )
    #
    # results_pd = model.predict(future_pd)
    #
    # results_pd = results_pd[['ds', 'yhat']]
    # results_pd['pdr'] = pdr_code
    #
    # return results_pd


if __name__ == "__main__":
    run()
