from src.api.mlflow_utils import load_model_from_registry


def test_load_model_from_registry_with_invalid_name():
    # This should handle failures gracefully and return None
    model = load_model_from_registry("non_existent_model_12345")
    assert model is None
