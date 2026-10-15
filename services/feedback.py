"""Turn analysts' case decisions into training data.

When an analyst resolves a case as "Confirmed fraud" or "Legitimate", that
verdict is a better label than anything the model guessed. `flask retrain`
adds these rows to the training data, so the next model learns from the
mistakes the last one made (see ml/train.py, add_reviewed_rows).
"""

import pandas as pd
from sqlalchemy import select

from ml.datasets import LABEL, get_profile
from models import DataSource, FraudCase, Transaction, db

VERDICTS = {"Confirmed fraud": 1, "Legitimate": 0}


def reviewed_cases(dataset):
    """Resolved cases of one format as cleaned rows, with the analyst's verdict in LABEL."""
    profile = get_profile(dataset)
    found = db.session.execute(
        select(Transaction.features, FraudCase.outcome)
        .join(FraudCase, FraudCase.transaction_id == Transaction.id)
        .join(DataSource, DataSource.id == Transaction.source_id)
        .where(DataSource.dataset == dataset, FraudCase.status == "Resolved",
               FraudCase.outcome.in_(list(VERDICTS)))
    ).all()
    rows = [features | {LABEL: VERDICTS[outcome]} for features, outcome in found
            if all(f in features for f in profile.field_names)]
    return pd.DataFrame(rows, columns=None if rows else profile.field_names + [LABEL])
