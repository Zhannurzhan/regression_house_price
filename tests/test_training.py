import math

import pandas as pd
import pytest
from joblib import load

from src.training_pipeline.eval import evaluate_model
from src.training_pipeline.train import train_model
from src.training_pipeline.tune import tune_model


@pytest.fixture()
def feature_data(tmp_path):
    train_path = tmp_path / "feature_engineered_train.csv"
    eval_path = tmp_path / "feature_engineered_eval.csv"

    train = pd.DataFrame({
        "year": [2018, 2018, 2019, 2019, 2019, 2018],
        "quarter": [1, 2, 1, 2, 3, 4],
        "month": [1, 4, 2, 5, 8, 11],
        "median_list_price": [200000, 220000, 240000, 260000, 280000, 300000],
        "zipcode_freq": [3, 3, 2, 2, 1, 1],
        "city_full_encoded": [210000, 210000, 250000, 250000, 290000, 290000],
        "price": [205000, 225000, 245000, 265000, 285000, 305000],
    })
    eval_df = pd.DataFrame({
        "year": [2020, 2020, 2021, 2021],
        "quarter": [1, 2, 1, 3],
        "month": [1, 6, 3, 9],
        "median_list_price": [310000, 330000, 350000, 370000],
        "zipcode_freq": [2, 2, 1, 1],
        "city_full_encoded": [300000, 320000, 340000, 360000],
        "price": [315000, 335000, 355000, 375000],
    })

    train.to_csv(train_path, index=False)
    eval_df.to_csv(eval_path, index=False)
    return train_path, eval_path


def _assert_metrics(metrics):
    assert set(metrics.keys()) == {"mae", "rmse", "r2"}
    assert all(isinstance(v, float) and math.isfinite(v) for v in metrics.values())


def test_train_creates_model_and_metrics(tmp_path, feature_data):
    train_path, eval_path = feature_data
    out_path = tmp_path / "xgb_model.pkl"

    _, metrics = train_model(
        train_path=train_path,
        eval_path=eval_path,
        model_output=out_path,
        model_params={"n_estimators": 20, "max_depth": 4, "learning_rate": 0.1},
    )

    assert out_path.exists()
    _assert_metrics(metrics)
    assert load(out_path) is not None


def test_eval_works_with_saved_model(tmp_path, feature_data):
    train_path, eval_path = feature_data
    model_path = tmp_path / "xgb_model.pkl"

    train_model(
        train_path=train_path,
        eval_path=eval_path,
        model_output=model_path,
        model_params={"n_estimators": 20},
    )

    metrics = evaluate_model(model_path=model_path, eval_path=eval_path)
    _assert_metrics(metrics)


def test_tune_saves_best_model(tmp_path, feature_data):
    train_path, eval_path = feature_data
    model_out = tmp_path / "xgb_best.pkl"
    tracking_uri = (tmp_path / "mlflow.db").as_uri().replace("file:///", "sqlite:///")

    best_params, best_metrics = tune_model(
        train_path=train_path,
        eval_path=eval_path,
        model_output=model_out,
        n_trials=2,
        tracking_uri=tracking_uri,
        experiment_name="test_xgb_optuna",
    )

    assert model_out.exists()
    assert isinstance(best_params, dict) and best_params
    _assert_metrics(best_metrics)
