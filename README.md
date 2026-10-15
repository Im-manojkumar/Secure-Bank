# SecureBank: AI-Powered Fraud Detection

SecureBank is a web application that imports bank transactions, scores them for fraud with a machine-learning model, and lets analysts review suspicious cases from one dashboard.

This repository is built in stages:

1. **Fraud model** (done): train and compare four classifiers on the Kaggle credit-card dataset.
2. **Flask backend and SQLite database** (done): CSV import, prediction API, login and roles.
3. Dashboard and case review: fraud statistics, transaction search, alerts and investigation notes.
4. Testing, security and documentation.

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

Download **Credit Card Fraud Detection** from Kaggle and save it as `data/raw/creditcard.csv`. The file is about 150 MB and is not committed to git.

It has 284,807 transactions, of which 492 are fraud. Columns are `Time`, `V1` to `V28` (anonymised features), `Amount` and `Class` (1 = fraud). It also contains 1,081 duplicate rows, which the loader removes.

To try the code without the real file, generate fake data with the same columns:

```bash
python -m ml.sample_data --rows 20000 --out data/raw/sample.csv
python -m ml.train --data data/raw/sample.csv
```

## Training the model

```bash
python -m ml.train
```

A run takes about 2 to 3 minutes on a laptop. It does the following:

1. Loads the CSV, checks the columns and drops duplicates.
2. Splits the data 60/20/20 into train, validation and test sets, keeping the same fraud rate in each.
3. Trains four pipelines, each of which scales the features, balances the classes with SMOTE (training data only) and fits a classifier:
   - Logistic Regression (baseline)
   - Decision Tree (readable rules)
   - Random Forest (ensemble)
   - Linear SVM (a kernel SVM would take hours on 170k rows)
4. Picks the model with the best **PR-AUC** on the validation set, then picks its alert threshold to maximise F1 on the validation set.
5. Scores the untouched test set once and writes the results to `ml/artifacts/`:
   - `model.joblib` (the saved pipeline, not committed) and `model_meta.json`
   - `metrics.json` and `comparison.txt`
   - `pr_curves.png` and `confusion_matrix.png`

Pass `--no-smote` to use class weights instead of SMOTE.

### Why not accuracy?

Only 0.17% of transactions are fraud, so a model that calls everything legitimate would still be 99.8% accurate. These metrics are reported instead:

- **Precision:** the share of alerts that really are fraud.
- **Recall:** the share of fraud that gets caught.
- **F1:** the balance between precision and recall.
- **PR-AUC:** precision and recall across every threshold. This is the main metric used to choose the model.
- **ROC-AUC:** reported for completeness.

## Results on creditcard.csv

These are test-set results (56,746 transactions, 95 of them fraud) from `ml/artifacts/comparison.txt`:

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | Missed fraud | False alerts |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.888 | 0.747 | 0.811 | 0.734 | 0.968 | 24 | 9 |
| Decision Tree | 0.602 | 0.747 | 0.667 | 0.476 | 0.927 | 24 | 47 |
| **Random Forest** | 0.885 | 0.726 | 0.798 | **0.788** | 0.987 | 26 | 9 |
| SVM (linear) | 0.755 | 0.747 | 0.751 | 0.707 | 0.970 | 24 | 23 |

Random Forest had the highest validation PR-AUC, so it is the saved model. At its chosen threshold it catches about 73% of fraud, and about 88% of its alerts are real fraud.

![Precision-recall curves](ml/artifacts/pr_curves.png)

## Scoring new transactions

```python
import pandas as pd
from ml.predict import FraudModel

model = FraudModel()
scores, suspicious = model.score(pd.read_csv("new_transactions.csv"))
```

A score is a ranking of how suspicious a transaction is, not a calibrated probability of fraud.

## Running the backend

The backend is a JSON API. The dashboard pages come in step 3.

```bash
export SECRET_KEY=change-me                       # Windows: set SECRET_KEY=change-me
python -m ml.train                                # once, to create the model
flask --app app init-db
flask --app app create-user admin@securebank.local --role admin
flask --app app import-csv data/raw/creditcard.csv
flask --app app run
```

Importing the full Kaggle file takes about a minute. The resulting database in `instance/securebank.db` is about 350 MB and is not committed. Importing the same rows again skips them as duplicates.

There are two roles. An **admin** can do everything, including importing files. An **analyst** can search transactions, run predictions and review fraud cases.

### What happens on import

1. The file is checked for the required columns `Time`, `V1` to `V28` and `Amount`. `Class` is optional. A file missing a column is rejected, and the failed import is still recorded.
2. Rows with non-numeric values, a negative amount or a `Class` other than 0 or 1 are skipped and counted.
3. Duplicate rows are skipped, whether they repeat within the file or were imported before.
4. Every row is scored by the saved model. Each transaction, its prediction and, when it is flagged, an **Open** fraud case are stored in separate tables, so an analyst's finding never overwrites what the model said.
5. `Time` is turned into a date by counting seconds from 1 September 2013, which is when the dataset begins.

### API

Every endpoint except login needs a logged-in session cookie. Errors come back as `{"error": "..."}`.

| Method and path | Who | What it does |
|---|---|---|
| `POST /api/auth/login` | anyone | Log in with JSON `{"email", "password"}` |
| `POST /api/auth/logout` | logged in | Log out |
| `GET /api/auth/me` | logged in | The current user |
| `POST /api/transactions/import` | admin | Upload a CSV as form field `file` |
| `GET /api/sources` | logged in | Past imports and their results |
| `GET /api/transactions` | logged in | Search, with filters `source_id`, `date_from`, `date_to`, `min_amount`, `max_amount`, `prediction` (`suspicious` or `low_risk`) and `status` |
| `GET /api/transactions/<id>` | logged in | One transaction with its model inputs |
| `POST /api/predict` | logged in | Score one transaction sent as JSON with all 30 model inputs, without saving it |
| `GET /api/alerts` | logged in | Fraud cases with the highest risk first, optionally filtered with `?status=Open` |
| `GET /api/cases/<id>` | logged in | One fraud case |
| `PATCH /api/cases/<id>` | admin, analyst | Update `status` (`Open`, `Under Review` or `Resolved`), `review_notes` and `outcome` (`Confirmed fraud` or `Legitimate`, required to resolve a case) |
| `GET /api/dashboard/summary` | logged in | Totals, predicted fraud, cases by status and a daily trend |

List endpoints take `page` and `per_page` (at most 100).

Example with curl:

```bash
curl -c cookies.txt -H "Content-Type: application/json" \
  -d '{"email":"admin@securebank.local","password":"..."}' http://127.0.0.1:5000/api/auth/login
curl -b cookies.txt "http://127.0.0.1:5000/api/alerts?status=Open&per_page=5"
curl -b cookies.txt -X PATCH -H "Content-Type: application/json" \
  -d '{"status":"Under Review","review_notes":"Called the cardholder"}' http://127.0.0.1:5000/api/cases/1
```

## Tests

```bash
pytest
```

The tests use generated data and an in-memory database, so they run without the real dataset. They cover the model pipeline and every API endpoint, including login, roles, import validation, duplicate detection, filters and the case workflow.

## Project layout

```
app.py            Flask app factory and CLI commands (init-db, create-user, import-csv)
config.py         settings, overridable with environment variables
models.py         database tables
routes/           API endpoints: auth, transactions, alerts, dashboard
services/         import, prediction and case logic used by the routes
ml/
  preprocess.py   load, validate and split the dataset
  train.py        build, train and compare the four models
  evaluate.py     metrics, threshold choice and charts
  predict.py      load the saved model and score new rows
  sample_data.py  generate fake data with the same columns
  artifacts/      saved model, metrics and charts
data/raw/         creditcard.csv goes here (not committed)
tests/            pytest tests
instance/         local SQLite database (not committed)
```
