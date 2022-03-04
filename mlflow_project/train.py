"""
Train a simple Prophet model.
"""

import click
import warnings
import mlflow
import mlflow.prophet
import pandas as pd
from fbprophet import Prophet


@click.command(
    help="Trains a Prophet model."
         "The model and its metrics are logged with mlflow."
)
@click.argument("training_data")
def run(training_data: pd.DataFrame) -> None:
    warnings.filterwarnings("ignore")

    pdr_code = training_data.iloc[0].pdr

    with mlflow.start_run():
        model = Prophet(
            interval_width=0.95,
            growth='linear',
            daily_seasonality=False,
            weekly_seasonality=True,
            yearly_seasonality=True,
            seasonality_mode='multiplicative'
        )

        training_data = training_data.rename(columns={'date': "ds", 'volume_giorno': "y"})

        # fit the model
        model.fit(training_data)

        mlflow.prophet.log_model(model, "model")

        # # configure predictions
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
