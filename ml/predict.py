"""Load a saved model and score new transactions.

The Flask app calls these; they work on rows already cleaned by the
model's dataset profile (see ml/datasets.py).
"""

import json
from pathlib import Path

import joblib

from ml.datasets import DEFAULT_PROFILE, get_profile
from ml.explain import explain, fraud_scores
from ml.train import ARTIFACTS_DIR


class FraudModel:
    def __init__(self, artifacts_dir=ARTIFACTS_DIR / DEFAULT_PROFILE):
        artifacts_dir = Path(artifacts_dir)
        model_path = artifacts_dir / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"No trained model at {model_path}. Run `python -m ml.train` first.")
        self.pipeline = joblib.load(model_path)
        self.meta = json.loads((artifacts_dir / "model_meta.json").read_text())
        self.profile = get_profile(self.meta.get("dataset", DEFAULT_PROFILE))
        self.threshold = self.meta["threshold"]

    def _inputs(self, df):
        missing = [c for c in self.profile.required_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {', '.join(missing)}")
        return self.profile.features(df)

    def score(self, df):
        """Return (scores, is_suspicious) arrays for the rows of df."""
        scores = fraud_scores(self.pipeline, self._inputs(df))
        return scores, scores >= self.threshold

    def explain(self, df, top=3):
        """For each row, the inputs that raised its score most (see ml/explain.py)."""
        return explain(self.pipeline, self.profile, self._inputs(df), self.meta["baseline"], top)
