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

    def unit_test_model(
        self, model_version_details: ModelVersion, n: int, floor: int, cap: int
    ):
        """
        # TODO: Doc
        """
        # Transition to "staging" stage and archive the last one (if present)
        model_version = self.client.transition_model_version_stage(
            name=model_version_details.name,
            version=model_version_details.version,
            stage="staging",
            archive_existing_versions=True,
        )

        # TODO: Da generalizzare rispetto a prophet
        model, flavor = self.load_model(
            model_uri=f"models:/{model_version.name}/{model_version.current_stage}"
        )

        pred_config = model.make_future_dataframe(
            periods=n, freq="d", include_history=False
        )

        # Add floor and cap
        pred_config["floor"] = floor
        pred_config["cap"] = cap

        pred = model.predict(pred_config)

        # Check quality
        out = "OK" if len(pred) == n else "KO"
        logger.info(f"### Unit test result: {out}")

        return out

    def deploy(
        self,
        run_id: str,
        key_code: str,
        models_config: dict,
        n_unit_test: int,
        cap: int,
    ):
        # TODO: VA TOLTO CAP DA QUI?
        """
        # TODO: Doc
        """

        # TODO: Quando avremo n modelli trainati in n run dovremo passare il run id del vincitore
        # Register the model
        logger.info("### Registering the model")
        model_uri = f"runs:/{run_id}/model"
        model_details = self.register_model(
            model_uri=model_uri, model_name=key_code, timeout_s=10
        )

        # Set model flavor tag
        if "model_flavor" in models_config:
            logger.info("### Setting model flavor tag")
            self.set_model_tag(
                model_version_details=model_details,
                tag_key="model_flavor",
                tag_value=models_config.get("model_flavor", ""),
            )

        # Unit test the model
        logger.info("### Performing model unit test")
        unit_test_status = self.unit_test_model(
            model_version_details=model_details,
            n=n_unit_test,
            floor=models_config.get("floor", 0),
            cap=cap,
        )

        if unit_test_status == "OK":
            # Take the current staging model and promote it to production
            # Archive the "already in production" model
            # Return status code
            logger.info("### Deploying model to Production.")
            model_version = self.client.transition_model_version_stage(
                name=model_details.name,
                version=model_details.version,
                stage="production",
                archive_existing_versions=True,
            )

            out = "OK" if model_version.status == "READY" else "KO"
            logger.info(f"### Deploy result: {out}")
