# SecureBank: AI-Powered Fraud Detection

[![Tests](https://github.com/Im-manojkumar/Secure-Bank/actions/workflows/tests.yml/badge.svg)](https://github.com/Im-manojkumar/Secure-Bank/actions/workflows/tests.yml)

SecureBank is a web application that imports bank transactions, scores them for fraud with machine-learning models, explains why each suspicious transaction was flagged, and lets analysts review suspicious cases from one dashboard.

This repository is built in stages:

1. **Fraud model** (done): train and compare classifiers on the Kaggle credit-card dataset.
2. **Flask backend and SQLite database** (done): CSV import, prediction API, login and roles.
3. **Dashboard and case review** (done): fraud statistics, transaction search, alerts and investigation notes.
4. **Testing, security and documentation** (done): audit log, login limits, CSRF and CSP protection, automated tests on GitHub, and architecture and test-case docs.
5. **More datasets and explanations** (done): separate models for the Sparkov and PaySim datasets, gradient boosting as a fifth model, and the reasons behind each alert.

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Datasets

SecureBank understands three public fraud datasets. Each one has its own columns, so each gets its own model, and the app recognises a file's format from its columns. Save the files in `data/raw/`; none of them is committed to git.

| Format | Download | Rows (fraud) | What a row is | Currency |
|---|---|---|---|---|
| `creditcard` | Kaggle "Credit Card Fraud Detection", saved as `creditcard.csv` | 284,807 (492) | A European card payment from two days in September 2013. `V1` to `V28` are anonymised PCA features | EUR |
| `sparkov` | Kaggle "Credit Card Transactions Fraud Detection Dataset" (`fraudTrain.csv` and `fraudTest.csv`) | 1,852,394 (9,651) | A simulated US card payment from 2019 and 2020, with merchant category, time, customer age and locations | USD |
| `paysim` | Kaggle "Synthetic Financial Datasets For Fraud Detection", renamed to `paysim.csv` | 6,362,620 (8,213) | A simulated mobile-money transfer, with account balances before and after | none (shown as "units") |

The model inputs for each format are defined in `ml/datasets.py`:

- **creditcard:** `Time`, `V1` to `V28` and `Amount`, used as they are.
- **sparkov:** amount, hour of day, night-time, day of week, customer age, city population, distance between customer and merchant, and merchant category. Names, street, job, ZIP code, gender and the full card number are never stored or used; only the card's last four digits are kept. Gender is left out of the model so that it cannot treat men and women differently.
- **paysim:** transfer type, amount, the four balances, whether the transfer empties the account, and how far each balance fails to add up after the transfer.

To try the code without the real files, generate fake data in any format:

```bash
python -m ml.sample_data --dataset sparkov --rows 20000 --out data/raw/sample.csv
python -m ml.train --dataset sparkov --data data/raw/sample.csv
```

## Training the models

```bash
python -m ml.train                                          # creditcard, about 2 minutes
python -m ml.train --dataset sparkov --data data/raw/fraudTrain.csv --test-data data/raw/fraudTest.csv
python -m ml.train --dataset paysim --data data/raw/paysim.csv --max-train-rows 1000000
```

Sparkov and PaySim take about 5 to 10 minutes each and need a few GB of memory, mostly to load the files. Each run does the following:

1. Loads the CSV, checks the columns, skips invalid rows and drops duplicates.
2. Splits the data into train, validation and test sets with the same fraud rate in each. Credit card and PaySim are split 60/20/20. Sparkov comes with a later test file, so `fraudTrain.csv` is split 75/25 into train and validation and `fraudTest.csv` is the test set.
3. With `--max-train-rows`, keeps every fraud row in the training set but only a sample of the genuine ones. PaySim uses this to train on 1 million of its 3.8 million training rows. The validation and test sets keep their real mix.
4. Trains five pipelines, each of which scales the inputs, balances the classes with SMOTE (training data only) and fits a classifier:
   - Logistic Regression (baseline)
   - Decision Tree (readable rules)
   - Random Forest (ensemble)
   - Linear SVM (a kernel SVM would take hours on this many rows)
   - Gradient Boosting (trees that each correct the ones before)
5. Picks the model with the best **PR-AUC** on the validation set, then picks its alert threshold to maximise F1 on the validation set.
6. Scores the untouched test set once and writes the results to `ml/artifacts/<format>/`:
   - `model.joblib` (the saved pipeline, not committed) and `model_meta.json`
   - `metrics.json` and `comparison.txt`
   - `pr_curves.png` and `confusion_matrix.png`

Pass `--no-smote` to use class weights instead of SMOTE.

### Why not accuracy?

Fraud is under 1% of every dataset, so a model that calls everything legitimate would still be over 99% accurate. These metrics are reported instead:

- **Precision:** the share of alerts that really are fraud.
- **Recall:** the share of fraud that gets caught.
- **F1:** the balance between precision and recall.
- **PR-AUC:** precision and recall across every threshold. This is the main metric used to choose the model.
- **ROC-AUC:** reported for completeness.

## Results

These are test-set results from `ml/artifacts/<format>/comparison.txt`. The saved model for each format is in bold.

### Credit card (56,746 test transactions, 95 fraud)

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | Missed fraud | False alerts |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.887 | 0.747 | 0.811 | 0.734 | 0.968 | 24 | 9 |
| Decision Tree | 0.602 | 0.747 | 0.667 | 0.476 | 0.927 | 24 | 47 |
| Random Forest | 0.885 | 0.726 | 0.798 | 0.787 | 0.987 | 26 | 9 |
| SVM (linear) | 0.755 | 0.747 | 0.751 | 0.707 | 0.970 | 24 | 23 |
| **Gradient Boosting** | 0.925 | 0.779 | 0.846 | **0.858** | 0.978 | 21 | 6 |

Gradient Boosting replaces Random Forest, the model from step 1. It catches 74 of the 95 frauds instead of 69, with 6 false alerts instead of 9.

### Sparkov (555,719 test transactions, 2,145 fraud)

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | Missed fraud | False alerts |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.278 | 0.566 | 0.373 | 0.181 | 0.964 | 930 | 3,161 |
| Decision Tree | 0.387 | 0.676 | 0.492 | 0.344 | 0.983 | 695 | 2,300 |
| Random Forest | 0.843 | 0.700 | 0.765 | 0.813 | 0.995 | 644 | 280 |
| SVM (linear) | 0.351 | 0.631 | 0.451 | 0.209 | 0.947 | 792 | 2,504 |
| **Gradient Boosting** | 0.849 | 0.767 | 0.806 | **0.869** | 0.998 | 500 | 293 |

The test file covers the six months after the training file, so this is the most realistic of the three results: the model is tested on later transactions than it learned from. Linear models do badly here because fraud depends on combinations, such as a large online purchase late at night, that a single straight line cannot separate.

### PaySim (1,272,416 test transactions, 1,640 fraud)

| Model | Precision | Recall | F1 | PR-AUC | ROC-AUC | Missed fraud | False alerts |
|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.738 | 0.497 | 0.594 | 0.634 | 0.997 | 825 | 290 |
| Decision Tree | 0.990 | 0.998 | 0.994 | 0.988 | 0.999 | 4 | 16 |
| **Random Forest** | 1.000 | 0.998 | 0.999 | **0.999** | 1.000 | 4 | 0 |
| SVM (linear) | 0.661 | 0.490 | 0.562 | 0.597 | 0.997 | 837 | 412 |
| Gradient Boosting | 1.000 | 0.998 | 0.999 | 0.998 | 1.000 | 4 | 0 |

The near-perfect score is a quirk of the simulator, not proof of a perfect model. In PaySim, fraudulent transfers almost always empty the account and leave balances that do not add up, which real bank data would not give away so easily.

![Precision-recall curves for credit card](ml/artifacts/creditcard/pr_curves.png)

## Why a transaction was flagged

When a model flags a transaction, SecureBank also records which inputs raised its score most. It replaces each input in turn with the value of a typical genuine transaction (the median, or the most common category), scores the transaction again, and reports how much the score drops. The three inputs with the biggest drop are stored with the prediction and shown on the transaction, case and check pages. A real flagged Sparkov transaction reads:

1. **Amount:** $890.22 (typical $47.30) · +0.995 to the score
2. **Time of day:** 22:00 to 22:59 (typical 14:00 to 14:59) · +0.086 to the score
3. **Merchant category:** Other (online) (typical Fuel and transport) · +0.032 to the score

On real Sparkov fraud the most common reasons are the amount, then the time of day and the merchant category. The code is in `ml/explain.py`. The typical values come from the training data and are saved in `model_meta.json`.

## Scoring new transactions

```python
import pandas as pd
from ml.predict import FraudModel

model = FraudModel("ml/artifacts/sparkov")
rows = pd.read_csv("new_transactions.csv")
scores, suspicious = model.score(rows)
reasons = model.explain(rows[suspicious])
```

A score is a ranking of how suspicious a transaction is, not a calibrated probability of fraud.

## Running the app

```bash
export SECRET_KEY=change-me                       # Windows: set SECRET_KEY=change-me
python -m ml.train                                # once per format you will import
flask --app app init-db
flask --app app create-user admin@securebank.local --role admin
flask --app app import-csv data/raw/creditcard.csv
flask --app app run
```

Then open http://127.0.0.1:5000 and log in.

Import Sparkov's `fraudTest.csv` or `paysim.csv` the same way. Importing the credit card file takes about a minute and makes a database of about 350 MB in `instance/securebank.db`, which is not committed. Importing the same rows again skips them as duplicates.

A database created before step 5 lacks the new `dataset` and `explanation` columns. Delete `instance/securebank.db` and run `init-db` again, because `init-db` only creates missing tables, not missing columns.

There are two roles. An **admin** can do everything, including importing files. An **analyst** can search transactions, run predictions and review fraud cases.

### What happens on import

1. The format is recognised from the file's columns. A file missing a required column of its closest format is rejected with the names of the missing columns, and the failed import is still recorded. The import page lists the columns each format needs.
2. Rows with non-numeric values, a negative amount, an unknown category or date, or a fraud label other than 0 or 1 are skipped and counted.
3. Duplicate rows are skipped, whether they repeat within the file or were imported before.
4. Every row is scored by that format's model. Each transaction, its prediction and, when it is flagged, an **Open** fraud case are stored in separate tables, so an analyst's finding never overwrites what the model said. Flagged rows also store the three inputs that raised their score.
5. Credit card `Time` is turned into a date by counting seconds from 1 September 2013, when that dataset begins. PaySim's `step` counts hours from 1 January 2016, a made-up start because the simulator has no real dates. Sparkov has real dates.

### Pages

| Page | What it shows |
|---|---|
| Dashboard (`/`) | Totals, the fraud rate, cases waiting for review, flagged amounts per currency, transactions and suspicious transactions over time, cases by status, and the highest-risk open alerts, for all data or one format |
| Transactions (`/transactions`) | Search by format, date, amount, prediction, case status and import, with each result linking to a detail page with the model inputs and why it was flagged |
| Alerts (`/alerts`) | Fraud cases, highest risk first, with tabs for Open, Under review, Resolved and All |
| Case (`/cases/<id>`) | The flagged transaction, why it was flagged, and a review form for status, outcome and investigation notes |
| Check a transaction (`/predict`) | Pick a format, score any set of values and see what raised the score, or start from an imported transaction and change it |
| Import (`/import`, admins only) | Upload a CSV with a progress bar, and see the history of past imports |
| Audit log (`/audit`, admins only) | Every login, failed login, import, case change and new user, with filters |

![Dashboard](docs/screenshots/dashboard-light.png)

| Alerts | Case review | Check a transaction |
|---|---|---|
| ![Alerts](docs/screenshots/alerts.png) | ![Case review](docs/screenshots/case.png) | ![Check a transaction](docs/screenshots/predict.png) |

![Transaction detail with why it was flagged](docs/screenshots/transaction.png)

The pages are plain HTML, CSS and JavaScript with no build step and no external libraries. Each page fetches its data from the JSON API below, and the charts are drawn as SVG. Light and dark mode follow the system setting, and the layout works on phone screens.

### API

Every endpoint except login needs a logged-in session cookie. Errors come back as `{"error": "..."}`.

| Method and path | Who | What it does |
|---|---|---|
| `POST /api/auth/login` | anyone | Log in with JSON `{"email", "password"}` |
| `POST /api/auth/logout` | logged in | Log out |
| `GET /api/auth/me` | logged in | The current user |
| `POST /api/transactions/import` | admin | Upload a CSV as form field `file`; the format is detected from its columns |
| `GET /api/sources` | logged in | Past imports and their results |
| `GET /api/transactions` | logged in | Search, with filters `dataset`, `source_id`, `date_from`, `date_to`, `min_amount`, `max_amount`, `prediction` (`suspicious` or `low_risk`) and `status` |
| `GET /api/transactions/<id>` | logged in | One transaction with its model inputs and explanation |
| `GET /api/models` | logged in | Each format, its input fields and its trained model's threshold and test results |
| `POST /api/predict` | logged in | Score one transaction sent as JSON, without saving it. `dataset` picks the format (default `creditcard`); the other keys are that format's input fields. The answer includes the explanation |
| `GET /api/alerts` | logged in | Fraud cases with the highest risk first, optionally filtered with `?status=Open` and `?dataset=sparkov` |
| `GET /api/cases/<id>` | logged in | One fraud case |
| `PATCH /api/cases/<id>` | admin, analyst | Update `status` (`Open`, `Under Review` or `Resolved`), `review_notes` and `outcome` (`Confirmed fraud` or `Legitimate`, required to resolve a case) |
| `GET /api/audit` | admin | Audit log, newest first, with filters `action` and `subject` |
| `GET /api/dashboard/summary` | logged in | Totals, predicted fraud, flagged amounts per currency, cases by status and a trend by day, or by hour with `?granularity=hour`; `?dataset=` limits it to one format |

List endpoints take `page` and `per_page` (at most 100).

Example with curl:

```bash
curl -c cookies.txt -H "Content-Type: application/json" \
  -d '{"email":"admin@securebank.local","password":"..."}' http://127.0.0.1:5000/api/auth/login
curl -b cookies.txt "http://127.0.0.1:5000/api/alerts?status=Open&per_page=5"
curl -b cookies.txt -X PATCH -H "Content-Type: application/json" \
  -d '{"status":"Under Review","review_notes":"Called the cardholder"}' http://127.0.0.1:5000/api/cases/1
```

## Security

The main protections are:
- **Passwords:** stored as salted scrypt hashes.
- **Login limit:** 5 failed logins per email and IP address in 15 minutes.
- **Roles:** checked on every route.
- **Cross-site requests:** blocked by `SameSite` cookies and an `Origin` check.
- **Scripts and styles:** a strict Content-Security-Policy, with no inline scripts or styles on any page.
- **Audit log:** every sensitive action is recorded.

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md#security) lists each risk, how it is handled, and the known limits.

Before running anywhere other than your own machine:
- Set a long random `SECRET_KEY`. Without one, a random key is generated and logins reset on every restart.
- Serve the app over HTTPS and set `SESSION_COOKIE_SECURE=1`.
- Use a production server such as gunicorn rather than `flask run`.

## Tests

```bash
pytest
```

There are 92 tests, which run on GitHub for every push and pull request. They use generated data and an in-memory database, so they run without the real dataset. [docs/TEST_CASES.md](docs/TEST_CASES.md) lists every case with its expected result. They cover the model pipeline, every API endpoint (login, roles, import validation, duplicate detection, filters and the case workflow) every page (login redirects, admin-only pages and safe redirects after login), the audit log, the security protections, and each dataset format with its explanations.

## Project layout

```
app.py            Flask app factory and CLI commands (init-db, create-user, import-csv)
config.py         settings, overridable with environment variables
models.py         database tables
routes/           API endpoints (auth, transactions, alerts, dashboard) and HTML pages (pages.py)
templates/        Jinja page templates
static/           CSS and JavaScript for the pages
services/         import, prediction and case logic used by the routes
ml/
  datasets.py     the three formats: columns, cleaning and model inputs
  preprocess.py   load, validate and split a dataset
  train.py        build, train and compare the five models
  evaluate.py     metrics, threshold choice and charts
  explain.py      which inputs raised a transaction's score
  predict.py      load a saved model, score and explain new rows
  sample_data.py  generate fake data in each format
  artifacts/      one folder per format with its model, metrics and charts
docs/             architecture, test cases and screenshots
data/raw/         the dataset CSVs go here (not committed)
tests/            pytest tests
instance/         local SQLite database (not committed)
```
