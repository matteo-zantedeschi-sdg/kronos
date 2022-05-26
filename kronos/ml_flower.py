from mlflow.tracking import MlflowClient
from mlflow.entities.model_registry import ModelVersion
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from typing import Optional
import mlflow
import time
import logging

logger = logging.getLogger(__name__)


class MLFlower:
    def __init__(self, client: MlflowClient):
        self.client = client
        self.experiment = Optional[str]

    def start_run(self, run_name: str):
        """
        # TODO: Doc
        """
        run = mlflow.start_run(experiment_id=self.experiment, run_name=run_name)
        logger.info(f"### Run started: {run}")
        return run

    @staticmethod
    def end_run():
        """
        # TODO: Doc
        """
        mlflow.end_run()
        logger.info("### Run ended")

    @staticmethod
    def load_model(model_uri: str):
        """
        # TODO: Doc
        """
        try:
            # Get PyFuncModel
            pyfunc_model = mlflow.pyfunc.load_model(model_uri)

            # Get loader module
            loader_module = pyfunc_model.metadata.flavors.get("python_function").get(
                "loader_module"
            )
            flavor = loader_module.split(".")[-1]

            # TODO: To add all the supported model classes
            if loader_module == "mlflow.prophet":
                model = mlflow.prophet.load_model(model_uri)
            elif loader_module == "mlflow.pmdarima":
                model = mlflow.pmdarima.load_model(model_uri)
            else:
                raise Exception("Model flavor not supported")

            return model, flavor

        except Exception as e:
            logger.error(f"### Model loading failed: {e}")

    def get_experiment(self, experiment_path: str):
        """
        # TODO:
        """
        # Create/Get experiment
        try:
            self.experiment = self.client.create_experiment(experiment_path)
            logger.info(f"### Experiment created: {self.experiment}")
        except Exception as e:
            logger.warning(f"### Experiment creation failed: {e}")
            self.experiment = self.client.get_experiment_by_name(
                experiment_path
            ).experiment_id
            logger.info(f"### Experiment retrieved: {self.experiment}")

    def register_model(
        self, model_uri: str, model_name: str, timeout_s: int
    ) -> ModelVersion:

        # Register the trained model to MLflow Registry
        model_details = mlflow.register_model(model_uri=model_uri, name=model_name)

        # Check registration Status
        for _ in range(timeout_s):
            model_version_details = self.client.get_model_version(
                name=model_details.name, version=model_details.version
            )
            status = ModelVersionStatus.from_string(model_version_details.status)
            if status == ModelVersionStatus.READY:
                break
            time.sleep(1)

        logger.info(f"### Model registered: {model_details}")

        return model_details

    def set_model_tag(
        self, model_version_details: ModelVersion, tag_key: str, tag_value: str
    ):
        """
        TODO: Doc
        """
        # Set the model flavor tag
        self.client.set_model_version_tag(
            name=model_version_details.name,
            version=model_version_details.version,
            key=tag_key,
            value=tag_value,
        )
        logger.info(f"### Tag set: {tag_key} - {tag_value}")

    def promote_model(
        self,
        model_version_details: ModelVersion,
        stage: str,
        archive_existing_versions: bool,
    ):
        """
        # TODO: Doc
        :return:
        """
        try:
            model_version = self.client.transition_model_version_stage(
                name=model_version_details.name,
                version=model_version_details.version,
                stage=stage,
                archive_existing_versions=archive_existing_versions,
            )
            return model_version
        except Exception as e:
            logger.error(f"Model promotion failed: {e}")
