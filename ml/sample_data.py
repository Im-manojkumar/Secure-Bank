"""Generate fake datasets with the same columns as the real ones.

Used by the tests and for trying the pipeline without the real files.
The numbers are random with a planted fraud pattern; results on them say
nothing about real fraud.

    python -m ml.sample_data --rows 20000 --out data/raw/sample.csv
    python -m ml.sample_data --dataset sparkov --rows 20000 --out data/raw/sparkov_sample.csv
    python -m ml.sample_data --dataset paysim --rows 20000 --out data/raw/paysim_sample.csv
"""

import argparse

import numpy as np
import pandas as pd

from ml.datasets import PAYSIM_TYPES, SPARKOV_CATEGORIES
from ml.preprocess import FEATURES, TARGET


def make_sample(rows=20000, fraud_rate=0.005, random_state=0):
    """Fake Kaggle credit-card rows: Time, V1..V28, Amount, Class."""
    rng = np.random.default_rng(random_state)
    y = (rng.random(rows) < fraud_rate).astype(int)
    v = rng.normal(size=(rows, 28))
    # Shift a few components for fraud rows so there is a pattern to learn.
    v[y == 1, :5] += rng.normal(2.5, 1.0, size=(int(y.sum()), 5))
    df = pd.DataFrame(v, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", np.sort(rng.uniform(0, 172800, rows)).round())
    df["Amount"] = np.round(rng.lognormal(3.5, 1.2, rows), 2)
    df[TARGET] = y
    return df[FEATURES + [TARGET]]


def make_sparkov_sample(rows=20000, fraud_rate=0.01, random_state=0):
    """Fake Sparkov rows. Fraud is planted as large online purchases late at night."""
    rng = np.random.default_rng(random_state)
    y = (rng.random(rows) < fraud_rate).astype(int)
    start = pd.Timestamp("2019-01-01")
    seconds = np.sort(rng.integers(0, 365 * 86400, rows))
    when = start + pd.to_timedelta(seconds, unit="s")
    night = start.normalize() + pd.to_timedelta(rng.integers(0, 365, rows), unit="D") \
        + pd.to_timedelta(rng.integers(22 * 3600, 26 * 3600, rows), unit="s")
    when = when.where(y == 0, night)
    categories = np.array(list(SPARKOV_CATEGORIES))
    category = rng.choice(categories, rows)
    category[y == 1] = rng.choice(["shopping_net", "grocery_pos", "misc_net"], int(y.sum()))
    amount = np.round(rng.lognormal(3.6, 1.0, rows), 2)
    amount[y == 1] = np.round(rng.uniform(250, 1300, int(y.sum())), 2)
    lat = rng.uniform(25, 48, rows)
    long = rng.uniform(-120, -70, rows)
    dob = pd.Timestamp("1950-01-01") + pd.to_timedelta(rng.integers(0, 50 * 365, rows), unit="D")
    return pd.DataFrame({
        "trans_date_trans_time": when.strftime("%Y-%m-%d %H:%M:%S"),
        # 200 cards, so each card has a history of earlier payments.
        "cc_num": rng.choice(rng.integers(10**15, 10**16, 200), rows),
        "merchant": [f"fraud_Shop {i % 500}" for i in range(rows)],
        "category": category,
        "amt": amount,
        "first": "Test", "last": "Person",
        "gender": rng.choice(["F", "M"], rows),
        "street": "1 Test Street",
        "city": "Springfield", "state": "IL", "zip": 62701,
        "lat": lat.round(4), "long": long.round(4),
        "city_pop": rng.integers(100, 2_000_000, rows),
        "job": "Tester",
        "dob": dob.strftime("%Y-%m-%d"),
        "trans_num": [f"{random_state:04d}{i:012d}" for i in range(rows)],
        "unix_time": seconds,
        "merch_lat": (lat + rng.normal(0, 0.5, rows)).round(4),
        "merch_long": (long + rng.normal(0, 0.5, rows)).round(4),
        "is_fraud": y,
    })


def make_paysim_sample(rows=20000, fraud_rate=0.01, random_state=0):
    """Fake PaySim rows. Fraud is planted as transfers and cash-outs that empty the sender's account."""
    rng = np.random.default_rng(random_state)
    y = (rng.random(rows) < fraud_rate).astype(int)
    kind = rng.choice(list(PAYSIM_TYPES), rows)
    kind[y == 1] = rng.choice(["TRANSFER", "CASH_OUT"], int(y.sum()))
    old_orig = np.round(rng.lognormal(10, 1.5, rows), 2)
    amount = np.round(old_orig * rng.uniform(0.01, 0.5, rows), 2)
    amount[y == 1] = old_orig[y == 1]
    new_orig = np.round(np.clip(old_orig - amount, 0, None), 2)
    old_dest = np.round(rng.lognormal(10, 2, rows), 2)
    new_dest = np.round(old_dest + amount, 2)
    new_dest[y == 1] = old_dest[y == 1]
    return pd.DataFrame({
        "step": np.sort(rng.integers(1, 745, rows)),
        "type": kind,
        "amount": amount,
        "nameOrig": [f"C{i:09d}" for i in range(rows)],
        "oldbalanceOrg": old_orig,
        "newbalanceOrig": new_orig,
        "nameDest": [f"M{i:09d}" for i in rng.integers(0, 10**9, rows)],
        "oldbalanceDest": old_dest,
        "newbalanceDest": new_dest,
        "isFraud": y,
        "isFlaggedFraud": 0,
    })


MAKERS = {"creditcard": make_sample, "sparkov": make_sparkov_sample, "paysim": make_paysim_sample}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="creditcard", choices=sorted(MAKERS))
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--out", default="data/raw/sample.csv")
    args = parser.parse_args()
    MAKERS[args.dataset](args.rows).to_csv(args.out, index=False)
    print(f"Wrote {args.rows:,} fake {args.dataset} rows to {args.out}")


if __name__ == "__main__":
    main()
