"""Spot when new transactions stop looking like the ones the model learned from.

At training time, the spread of each input group and of the model's scores on
the validation set is saved with the model. Each import is compared with it
using the population stability index (PSI), a standard check in credit
scoring:

    PSI = sum over bins of (new share - reference share) * ln(new share / reference share)

Below 0.1 the data is stable; 0.1 to 0.25 is a moderate shift worth watching;
above 0.25 is a major shift, and the model's scores may no longer be reliable.
"""

import numpy as np

BINS = 10
MODERATE, MAJOR = 0.1, 0.25
# Fewer rows than this give PSI values too noisy to act on.
MIN_ROWS = 200
# Empty bins would make PSI infinite, so every share is at least this.
FLOOR = 1e-4


def _numeric_reference(values):
    edges = np.unique(np.quantile(values, np.linspace(0, 1, BINS + 1)[1:-1]))
    return {"edges": edges.tolist(), "shares": _shares(values, edges).tolist()}


def _shares(values, edges):
    counts = np.bincount(np.searchsorted(edges, values, side="right"), minlength=len(edges) + 1)
    return counts / max(len(values), 1)


def build_reference(profile, X, scores):
    """The reference spreads saved in model_meta.json: one per input group, plus the scores."""
    groups = {}
    for group in profile.groups:
        if not group.drift:
            continue
        if group.onehot:
            groups[group.label] = {"columns": group.columns,
                                   "shares": X[group.columns].mean().round(6).tolist()}
        else:
            groups[group.label] = {"column": group.columns[0]} | _numeric_reference(X[group.columns[0]].to_numpy())
    return {"score": _numeric_reference(np.asarray(scores)), "inputs": groups}


def psi(expected, actual):
    expected = np.clip(np.asarray(expected, dtype=float), FLOOR, None)
    actual = np.clip(np.asarray(actual, dtype=float), FLOOR, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def level(value):
    return "major" if value >= MAJOR else "moderate" if value >= MODERATE else "stable"


def check(reference, X, scores, top=3):
    """Compare new rows with the reference. Returns a summary dict, or None if too few rows."""
    if reference is None or len(X) < MIN_ROWS:
        return None
    score_ref = reference["score"]
    score_psi = psi(score_ref["shares"], _shares(np.asarray(scores), np.asarray(score_ref["edges"])))
    inputs = []
    for label, ref in reference["inputs"].items():
        if "columns" in ref:
            actual = X[ref["columns"]].mean().to_numpy()
        else:
            actual = _shares(X[ref["column"]].to_numpy(), np.asarray(ref["edges"]))
        inputs.append({"input": label, "psi": round(psi(ref["shares"], actual), 4)})
    inputs.sort(key=lambda item: -item["psi"])
    worst = max([score_psi] + [i["psi"] for i in inputs])
    return {
        "rows": len(X),
        "level": level(worst),
        "score_psi": round(score_psi, 4),
        "most_shifted": [i | {"level": level(i["psi"])} for i in inputs[:top]],
    }


def describe(result):
    """One sentence for the import message, or "" when there is nothing to say."""
    if result is None or result["level"] == "stable":
        return ""
    shifted = ", ".join(f"{i['input']} (PSI {i['psi']:.2f})" for i in result["most_shifted"] if i["level"] != "stable")
    scores = f"model scores (PSI {result['score_psi']:.2f})" if result["score_psi"] >= MODERATE else ""
    what = ", ".join(x for x in (shifted, scores) if x)
    if result["level"] == "major":
        return f"Warning: these transactions look very different from the training data in {what}; scores may be unreliable, so consider retraining."
    return f"Note: these transactions differ somewhat from the training data in {what}."
