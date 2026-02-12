import logging
from typing import Optional

logger = logging.getLogger(__name__)


def load_model_from_registry(model_name: str, stage: str = "Production") -> Optional[object]:
    """Attempt to load a model from the MLflow Model Registry.

    Returns the loaded model or None if loading fails (e.g., mlflow not installed or no MLflow server configured).
    """
    try:
        import mlflow
    except Exception as exc:
        logger.warning(f"mlflow is not available: {exc}")
        return None

    model_uri = f"models:/{model_name}/{stage}"
    try:
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Loaded model {model_name} from MLflow stage={stage}")
        return model
    except Exception as exc:
        logger.warning(f"Could not load model from MLflow ({model_uri}): {exc}")
        return None
