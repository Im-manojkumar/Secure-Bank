"""Load the saved model and score new transactions.

The Flask app will call these functions; they work on any DataFrame that
has the training feature columns.
"""

import json
from pathlib import Path

import joblib

from ml.preprocess import FEATURES
from ml.train import ARTIFACTS_DIR, fraud_scores


class FraudModel:
    def __init__(self, artifacts_dir=ARTIFACTS_DIR):
        artifacts_dir = Path(artifacts_dir)
        model_path = artifacts_dir / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"No trained model at {model_path}. Run `python -m ml.train` first.")
        self.pipeline = joblib.load(model_path)
        self.meta = json.loads((artifacts_dir / "model_meta.json").read_text())
        self.threshold = self.meta["threshold"]

    def score(self, df):
        """Return (scores, is_suspicious) arrays for the rows of df."""
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {', '.join(missing)}")
        scores = fraud_scores(self.pipeline, df[FEATURES])
        return scores, scores >= self.threshold
