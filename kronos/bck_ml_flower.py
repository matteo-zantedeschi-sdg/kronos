from mlflow.tracking import MlflowClient
from mlflow.entities import Experiment
from mlflow.entities.model_registry import ModelVersion
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
import mlflow
import time


class MLFlower:
    def __init__(self):

        # Init MLflow client
        self.client = MlflowClient()

    def get_experiment(self, experiment_name: str) -> Experiment:

        # Search for specific experiment
        experiment = self.client.get_experiment_by_name(experiment_name)

        if experiment is None:
            experiment = self.client.create_experiment(experiment_name)
            # client.set_experiment_tag(experiment_id, 'env', 'test') (TODO)
            print("Experiment created")
        else:
            print("Experiment retrieved")

        return experiment

    def register_model(
        self, model_uri: str, model_name: str, timeout_s: int, model_flavor_tag: str
    ) -> ModelVersion:
        # Add the current new model in the model registry and add it in staging
        # Return status code
        # E.g. model_details = mlflow.register_model(model_uri=f"runs:/{last_run.info.run_uuid}/model", model_name="03081000000640", timeout_s=10)

        # Register the model
        model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

        # Check Status
        for _ in range(timeout_s):
            model_version_details = self.client.get_model_version(
                name=model_details.name, version=model_details.version
            )
            status = ModelVersionStatus.from_string(model_version_details.status)
            if status == ModelVersionStatus.READY:
                break
            time.sleep(1)

        # Set the flavor tag
        self.client.set_model_version_tag(
            name=model_version_details.name,
            version=model_version_details.version,
            key="model_flavor",
            value=model_flavor_tag,
        )

        # Add in Staging and archive the last one (if present)
        model_version = self.client.transition_model_version_stage(
            name=model_version_details.name,
            version=model_version_details.version,
            stage="staging",
            archive_existing_versions=True,
        )

        if model_version.status == "READY":
            return model_version
        else:
            return None

    def retrieve_model_version(self, name: str, stage: str) -> ModelVersion:

        # Take all model versions
        model_versions = self.client.search_model_versions(f"name='{name}'")

        # Take model versions belonging to a speicific stage
        stage_models = [
            model for model in model_versions if model.current_stage == stage
        ]

        # More than one model in that stage
        if len(stage_models) > 1:
            print(
                f"WARNING - More than one model has been found in {stage} - should be just one!"
            )
            model = sorted(stage_models, key=lambda x: x.version, reverse=True)[0]
        # No model in that stage
        elif len(stage_models) < 1:
            print(f"ERROR - No model has been found in {stage}!")
            model = None
        # Exactly one model in that stage
        else:
            model = stage_models[0]

        return model

    def unit_test_model(self, model_version: ModelVersion, unit_test_days: int) -> str:
        # Load the current staging model and test its prediction method

        # Retrieve model flavor tag
        # model_flavor_tag = model_version.tags['model_flavor']

        # Unit test
        # if model_flavor_tag == 'prophet':
        # Retrieve model and make predictions
        model = mlflow.prophet.load_model(
            f"models:/{model_version.name}/{model_version.current_stage}"
        )
        pred_conf = model.make_future_dataframe(
            periods=unit_test_days, freq="d", include_history=False
        )
        pred = model.predict(pred_conf)
        # else:
        # print(f"Model flavor {model_flavor_tag} not managed.")
        # pred = None

        # Check quality
        out = "OK" if len(pred) == unit_test_days else "KO"

        return out

    def deploy_model(self, model_version: ModelVersion) -> str:
        # Take the current staging model and promote it to production
        # Archive the "already in production" model
        # Return status code

        model_version = self.client.transition_model_version_stage(
            name=model_version.name,
            version=model_version.version,
            stage="production",
            archive_existing_versions=True,
        )

        out = "OK" if model_version.status == "READY" else "KO"

        return out
