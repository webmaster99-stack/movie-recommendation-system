from recsys.utils.config import load_params


def test_params_has_required_sections() -> None:
    params = load_params()
    assert isinstance(params["seed"], int)
    assert params["runtime"]["num_threads"] >= 1
    assert params["mlflow"]["experiment_name"]
