import logging
import time
from typing import Optional

import mlflow
from mlflow.entities.model_registry import ModelVersion
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


class MLFlower:
    """
    Class to manage all mlflow activities.
    """

    def __init__(self, client: MlflowClient) -> None:
        """
        Initialization method.

        :param MlflowClient client: The mlflow client used to track experiments and register models.

        :return: No return.

        **Example**

        .. code-block:: python

            ml_flower = MLFlower(client=client)

        """
        self.client = client
        self.experiment = Optional[str]

    def start_run(self, run_name: str) -> mlflow.ActiveRun:
        """
        Start a new MLflow run with a specific run name.

        :param str run_name: Name of the run to start.

        :return: *(mlflow.ActiveRun)* object that acts as a context manager wrapping the run’s state.
        """
        try:
            run = mlflow.start_run(experiment_id=self.experiment, run_name=run_name)
            logger.info(f"### Run started: {run}")
            return run

        except Exception as e:
            logger.error(f"### Start of the run {run_name} failed: {e}")

    @staticmethod
    def end_run() -> None:
        """
        A static method to end an active MLflow run (if there is one).

        :return: No return.
        """
        try:
            mlflow.end_run()
            logger.info("### Run ended")

        except Exception as e:
            logger.error(f"### End of the run failed: {e}")

    def get_parameters(self, model_uri) -> dict:
        import json

        try:
            pyfunc_model = mlflow.pyfunc.load_model(model_uri)
            res = self.client.get_run(pyfunc_model.metadata.run_id).data.params
            for k, val in res.items():
                try:
                    res[k] = json.loads(val.lower())
                except ValueError:
                    res[k] = val
            return res
        except Exception as e:
            logger.error(f"### Parameters loading failed: {e}")

    @staticmethod
    def load_model(model_uri: str):
        """
        Static method to load a model in Python function format from a specific uri.
        Based on its flavor (e.g. 'prophet') the model is later reloaded using the specific loader module.

        :param str model_uri: The uri of the model to load.

        :return:
            * The model flavor detected.
            * The model loaded.
        """
        try:
            # Get PyFuncModel
            pyfunc_model = mlflow.pyfunc.load_model(model_uri)

            # Get loader module
            loader_module = pyfunc_model.metadata.flavors.get("python_function").get(
                "loader_module"
            )
            flavor = loader_module.split(".")[-1]

            if loader_module == "mlflow.prophet":
                model = mlflow.prophet.load_model(model_uri)
            elif loader_module == "mlflow.pmdarima":
                model = mlflow.pmdarima.load_model(model_uri)
            elif loader_module == "mlflow.tensorflow":
                model = mlflow.tensorflow.load_model(model_uri)
            else:
                raise Exception("Model flavor not supported")

            return model, flavor

        except Exception as e:
            logger.error(f"### Model loading failed: {e}")

    def get_experiment(self, experiment_path: str) -> None:
        """
        Method that try to create an experiment in a specific experiment path.
        If the experiment already exists, then the experiment is retrieved.

        :param str experiment_path: The experiment path to create/get.

        :return: No return.
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
        """
        Create a new model version in model registry for the model files specified by model_uri.

        :param str model_uri: Uri of the model to register.
        :param str model_name: Name of the registered model under which to create a new model version.
        :param int timeout_s: Maximum number of seconds to wait for the model version to finish being created.

        :return: *(ModelVersion)* The ModelVersion object created, corresponding to the model registered.
        """
        try:
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

        except Exception as e:
            logger.error(
                f"### Registration of the model {model_name} with uri {model_uri} failed: {e}"
            )

    def set_model_tag(
        self, model_version_details: ModelVersion, tag_key: str, tag_value: str
    ) -> None:
        """
        Set a tag for a specific model version.

        :param ModelVersion model_version_details: Registered model version details.
        :param str tag_key: Tag key to log.
        :param str tag_value: Tag value to log.

        :return: No return.
        """
        try:
            # Set the model flavor tag
            self.client.set_model_version_tag(
                name=model_version_details.name,
                version=model_version_details.version,
                key=tag_key,
                value=tag_value,
            )
            logger.info(f"### Tag set: {tag_key} - {tag_value}")

        except Exception as e:
            logger.error(
                f"### Setting tag {tag_key}-{tag_value} on model {model_version_details} failed: {e}"
            )

    def promote_model(
        self,
        model_version_details: ModelVersion,
        stage: str,
        archive_existing_versions: bool,
    ) -> ModelVersion:
        """
        Method used to update the model version stage.

        :param ModelVersion model_version_details: Model version details of the model to promote.
        :param str stage: Stage in which to promote the model.
        :param bool archive_existing_versions: If this flag is set to True, all existing model versions in the stage will be automically moved to the “archived” stage.

        :return: *(ModelVersion)* Model version object promoted.
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
            logger.error(f"### Model promotion failed: {e}")
