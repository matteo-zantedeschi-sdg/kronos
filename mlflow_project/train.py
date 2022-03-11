"""
Train a simple Prophet model.
"""

import click
import warnings
import mlflow
import mlflow.prophet
import pandas as pd
import copy
import time
from prophet import Prophet
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from modeler import Modeler
from ml_flower import MLFlower


@click.command(help="Trains a Prophet model." "The model and its metrics are logged with mlflow.")
@click.argument("data")
@click.argument("run_id")
def run(data: str, run_id: str) -> None:

    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Convert received data as Pandas DataFrame
    data = pd.read_json(data)

    # Read pdr code
    pdr_code = data['pdr'].iloc[0]

    # Log pdr as param
    mlflow.log_param('pdr', pdr_code)

    # Define all parameters - TODO: Dovranno essere passati dall'esterno
    n_test = 7
    unit_test_days = 7
    forecast_horizon = 7
    date_col = 'date'
    key_col = 'pdr'
    metric_col = 'volume_giorno'
    model_flavor = 'prophet'
    interval_width = 0.95
    growth = 'linear'
    daily_seasonality = False
    weekly_seasonality = True
    yearly_seasonality = True
    seasonality_mode = 'multiplicative'

    # Prophet specific pre-processing
    data = data.rename(columns={date_col: "ds", metric_col: "y"})

    # Train/Test split
    print("Train/Test split")
    train_data, test_data = Modeler.train_test_split(data=data, date_column="ds", n_test=n_test)

    # Specify model flavor tag
    mlflow.set_tag(key='model_flavor', value=model_flavor)

    # Log all params: in general, log all params different from default
    # TODO: Esplorare auto-log
    print("Logging parameters")
    mlflow.log_param("interval_width", interval_width)
    mlflow.log_param("growth", growth)
    mlflow.log_param("daily_seasonality", daily_seasonality)
    mlflow.log_param("weekly_seasonality", weekly_seasonality)
    mlflow.log_param("yearly_seasonality", yearly_seasonality)
    mlflow.log_param("seasonality_mode", seasonality_mode)

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
    model.fit(train_data)

    # Log the model
    print("Log the model")
    signature = infer_signature(train_data, model.predict(train_data))
    mlflow.prophet.log_model(pr_model=model, artifact_path='model', signature=signature)

    # Configure prediction
    pred_conf = model.make_future_dataframe(periods=n_test, freq='d', include_history=False)

    # Make predictions
    pred = model.predict(pred_conf)

    # Combine pred with original test data
    pred = pred[['ds', 'yhat']].set_index('ds')
    test_train_pred = copy.deepcopy(test_data)[['ds', 'y']].set_index('ds')
    test_train_pred = test_train_pred.join(pred, how='left').reset_index(level=0, inplace=True)

    # Check if a production model already exist
    prod_model_fl = False
    try:
        prod_model = mlflow.prophet.load_model(f"models:/{pdr_code}/Production")
        prod_model_fl = True
    except Exception as e:
        print(e)

    # Predict with current production model
    if prod_model_fl:
        prod_model_pred_conf = prod_model.make_future_dataframe(periods=n_test, freq='d', include_history=False)
        prod_pred = prod_model.predict(prod_model_pred_conf)
        prod_pred = prod_pred[['ds', 'yhat']].set_index('ds')
        test_prod_pred = copy.deepcopy(test_data)[['ds', 'y']].set_index('ds')
        test_prod_pred = test_prod_pred.join(prod_pred, how='left').reset_index(level=0, inplace=True)

    # Evaluate trained model
    # train_rmse = Modeler.evaluate_model(data=test_train_pred, metric='rmse', predicted_col='yhat', true_value_col='y')
    # mlflow.log_metric("rmse", train_rmse)

    # Evaluate current production model
    # if prod_model_fl:
        # prod_rmse = Modeler.evaluate_model(data=test_prod_pred, metric='rmse', predicted_col='yhat', true_value_col='y')

        # Competition
        # Combine all metrics
        # TODO: Now is just rmse, later could be a combination of all metrics
        # trained_model_score = -train_rmse
        # prod_model_score = -prod_rmse

    # Compare score
    if prod_model_fl:  # and prod_model_score > trained_model_score:
        print("Prod model is still winning.")
    else:
        print("New trained model is better than previous production model.")

        # Register model

        # Init MLflow client
        client = MlflowClient()
        model_details = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=str(pdr_code))

        # Check Status
        for _ in range(10):
            model_version_details = client.get_model_version(name=model_details.name, version=model_details.version)
            status = ModelVersionStatus.from_string(model_version_details.status)
            if status == ModelVersionStatus.READY:
                break
            time.sleep(1)

        # Set the flavor tag
        client.set_model_version_tag(
            name=model_version_details.name,
            version=model_version_details.version,
            key='model_flavor',
            value=model_flavor
        )

        # Add in Staging and archive the last one (if present)
        model_version = client.transition_model_version_stage(
            name=model_version_details.name,
            version=model_version_details.version, stage='staging',
            archive_existing_versions=True
        )

        # Unit test the model
        unit_test_status = MLFlower().unit_test_model(model_version=model_version, unit_test_days=unit_test_days)

        # Deploy model
        if unit_test_status == 'OK':
            deploy_status = MLFlower().deploy_model(model_version=model_version)


if __name__ == "__main__":
    run()
