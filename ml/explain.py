"""Say which inputs pushed a transaction's fraud score up.

The method is occlusion: for each group of inputs (say "Amount"), the row is
scored again with that group set to its value in a typical genuine
transaction. The drop in score is how much that group raised it. It needs no
extra library and works the same for every model type, at the cost of one
extra scoring pass per group.
"""

import numpy as np


def fraud_scores(pipeline, X):
    """Higher score = more suspicious. Uses probabilities when the model has them."""
    if hasattr(pipeline, "predict_proba"):
        return pipeline.predict_proba(X)[:, 1]
    return pipeline.decision_function(X)


def baseline_values(profile, X_genuine):
    """A typical genuine transaction: the median of each input, and the commonest
    option of each one-hot group. Saved with the model and used as the comparison point."""
    baseline = {c: float(v) for c, v in X_genuine.median().items()}
    for group in profile.groups:
        if group.onehot:
            commonest = X_genuine[group.columns].mean().idxmax()
            for c in group.columns:
                baseline[c] = 1.0 if c == commonest else 0.0
    return baseline


def explain(pipeline, profile, X, baseline, top=3):
    """For each row of X, up to `top` input groups that raised its score, largest first.

    Each item is {"feature", "value", "typical", "effect"}; effect is the score
    the group added compared with a typical genuine value (groups adding under 0.001 are left out).
    """
    if not len(X):
        return []
    groups = profile.groups
    base = fraud_scores(pipeline, X)
    effects = np.empty((len(X), len(groups)))
    for j, group in enumerate(groups):
        changed = X.copy()
        for c in group.columns:
            changed[c] = baseline[c]
        effects[:, j] = base - fraud_scores(pipeline, changed)

    typical = [g.describe(baseline) for g in groups]
    records = X.to_dict("records")
    results = []
    for i, values in enumerate(records):
        order = np.argsort(-effects[i])[:top]
        results.append([
            {
                "feature": groups[j].label,
                "value": groups[j].describe(values),
                "typical": typical[j],
                "effect": round(float(effects[i, j]), 4),
            }
            for j in order if effects[i, j] >= 0.001
        ])
    return results
