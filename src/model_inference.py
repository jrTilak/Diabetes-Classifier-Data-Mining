"""Load trained artifacts and prepare patient data for inference."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import joblib
import pandas as pd

from project_paths import MODEL_PATH, SCALER_PATH

FEATURE_NAMES = (
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
)

IMPUTE_MEDIANS = {
    "Glucose": 121.5,
    "BloodPressure": 72.0,
    "SkinThickness": 29.0,
    "Insulin": 125.0,
    "BMI": 32.3,
}


@dataclass(frozen=True)
class Prediction:
    """A binary prediction and its diabetes probability."""

    is_diabetic: bool
    diabetes_probability: float

    @property
    def non_diabetes_probability(self) -> float:
        return 1.0 - self.diabetes_probability


def load_artifacts() -> tuple[Any, Any]:
    """Load the trained classifier and its fitted feature scaler."""

    missing = [path for path in (MODEL_PATH, SCALER_PATH) if not path.is_file()]
    if missing:
        formatted_paths = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing model artifact(s): {formatted_paths}")

    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)


def prepare_features(values: Mapping[str, float]) -> pd.DataFrame:
    """Return one model-ready row in the feature order used during training."""

    missing_features = [name for name in FEATURE_NAMES if name not in values]
    if missing_features:
        raise ValueError(f"Missing feature(s): {', '.join(missing_features)}")

    processed: dict[str, float] = {}
    for name in FEATURE_NAMES:
        value = float(values[name])
        processed[name] = (
            IMPUTE_MEDIANS[name] if name in IMPUTE_MEDIANS and value == 0 else value
        )

    return pd.DataFrame([processed], columns=FEATURE_NAMES)


def predict(values: Mapping[str, float], model: Any, scaler: Any) -> Prediction:
    """Scale patient values and return the model prediction."""

    features = prepare_features(values)
    scaled_features = scaler.transform(features)
    predicted_class = int(model.predict(scaled_features)[0])
    diabetes_probability = float(model.predict_proba(scaled_features)[0, 1])

    return Prediction(
        is_diabetic=predicted_class == 1,
        diabetes_probability=diabetes_probability,
    )
