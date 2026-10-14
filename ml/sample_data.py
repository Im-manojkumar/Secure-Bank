"""Generate a fake dataset with the same columns as creditcard.csv.

Used by the tests and for trying the pipeline without the real file.
The numbers are random; results on it say nothing about real fraud.

    python -m ml.sample_data --rows 20000 --out data/raw/sample.csv
"""

import argparse

import numpy as np
import pandas as pd

from ml.preprocess import FEATURES, TARGET


def make_sample(rows=20000, fraud_rate=0.005, random_state=0):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--out", default="data/raw/sample.csv")
    args = parser.parse_args()
    make_sample(args.rows).to_csv(args.out, index=False)
    print(f"Wrote {args.rows:,} fake rows to {args.out}")


if __name__ == "__main__":
    main()
