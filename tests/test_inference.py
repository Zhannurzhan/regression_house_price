import sys
from pathlib import Path

import pandas as pd
import pytest
from category_encoders import TargetEncoder
from joblib import dump
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.inference_pipeline import inference


@pytest.fixture()
def inference_artifacts(tmp_path, monkeypatch):
    raw_train = pd.DataFrame({
        "date": ["2018-01-01", "2018-02-01", "2019-01-01", "2019-02-01"],
        "zipcode": [10001, 10001, 20002, 20002],
        "city_full": ["alpha", "alpha", "beta", "beta"],
        "median_list_price": [200000, 220000, 300000, 320000],
        "price": [205000, 225000, 305000, 325000],
    })

    freq_map = raw_train["zipcode"].value_counts()
    target_encoder = TargetEncoder(cols=["city_full"])
    target_encoder.fit(raw_train[["city_full"]], raw_train["price"])
    city_encoded = target_encoder.transform(raw_train[["city_full"]])["city_full"]

    feature_cols = [
        "year",
        "quarter",
        "month",
        "median_list_price",
        "zipcode_freq",
        "city_full_encoded",
    ]
    X_train = pd.DataFrame({
        "year": [2018, 2018, 2019, 2019],
        "quarter": [1, 1, 1, 1],
        "month": [1, 2, 1, 2],
        "median_list_price": raw_train["median_list_price"],
        "zipcode_freq": [2, 2, 2, 2],
        "city_full_encoded": city_encoded,
    })

    model = XGBRegressor(n_estimators=5, max_depth=2, random_state=42)
    model.fit(X_train, raw_train["price"])

    model_path = tmp_path / "xgb_model.pkl"
    freq_path = tmp_path / "freq_encoder.pkl"
    target_path = tmp_path / "target_encoder.pkl"
    dump(model, model_path)
    dump(freq_map, freq_path)
    dump(target_encoder, target_path)

    monkeypatch.setattr(inference, "TRAIN_FEATURE_COLUMNS", feature_cols)
    return raw_train, model_path, freq_path, target_path


def test_inference_runs_and_returns_predictions(inference_artifacts):
    sample_df, model_path, freq_path, target_path = inference_artifacts

    preds_df = inference.predict(
        sample_df,
        model_path=model_path,
        freq_encoder_path=freq_path,
        target_encoder_path=target_path,
    )

    assert not preds_df.empty
    assert "predicted_price" in preds_df.columns
    assert pd.api.types.is_numeric_dtype(preds_df["predicted_price"])
