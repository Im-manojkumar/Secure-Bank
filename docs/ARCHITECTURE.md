# SecureBank architecture

SecureBank is a single Flask application with a SQLite database and a scikit-learn model saved to disk. A browser loads HTML pages from Flask, and each page's JavaScript calls the app's JSON API.

## Components

```mermaid
flowchart LR
    subgraph Browser
        P[HTML pages + JavaScript<br/>templates/, static/]
    end

    subgraph Flask app
        R[Routes<br/>routes/*.py]
        S[Services<br/>ingestion, prediction,<br/>alerts, audit]
        M[Saved model<br/>ml/artifacts/model.joblib]
    end

    D[(SQLite<br/>instance/securebank.db)]
    CSV[CSV file<br/>upload or CLI]
    T[Training script<br/>python -m ml.train]
    K[(creditcard.csv)]

    P -- "fetch /api/... (session cookie)" --> R
    R --> S
    S --> D
    S -- score rows --> M
    CSV --> R
    K --> T -- writes --> M
```

| Layer | Files | Job |
|---|---|---|
| Pages | `templates/`, `static/js/`, `static/css/` | Show data and forms. They hold no business logic and only call the API. |
| Routes | `routes/*.py` | Check login and role, validate input, and return JSON or HTML. |
| Services | `services/*.py` | Import, scoring, case rules and audit logging, kept out of the routes so they can be tested and reused by the CLI. |
| Model | `ml/*.py`, `ml/artifacts/` | Training, evaluation and a `FraudModel` class that loads the saved pipeline. |
| Data | `models.py`, `instance/securebank.db` | Six tables, described below. |

## What happens when a file is imported

```mermaid
sequenceDiagram
    actor Admin
    participant Page as Import page
    participant API as POST /api/transactions/import
    participant Ingest as services/ingestion.py
    participant Model as FraudModel
    participant DB as SQLite

    Admin->>Page: choose creditcard.csv
    Page->>API: upload (multipart)
    API->>Ingest: import_transactions(file)
    Ingest->>Ingest: check columns, drop bad rows,<br/>hash rows to find duplicates
    Ingest->>DB: which hashes already exist?
    Ingest->>Model: score all new rows at once
    Ingest->>DB: insert transactions, predictions,<br/>and an Open case per flagged row
    Ingest->>DB: data_sources row + audit_logs row
    API-->>Page: rows imported, skipped, flagged
```

## Data model

```mermaid
erDiagram
    users ||--o{ data_sources : imports
    users ||--o{ fraud_cases : reviews
    users ||--o{ audit_logs : "acts in"
    data_sources ||--o{ transactions : contains
    transactions ||--|| predictions : "scored as"
    transactions ||--o| fraud_cases : "flagged as"

    users {
        int id
        string email
        string password_hash
        string role
    }
    data_sources {
        int id
        string name
        string status
        int rows_imported
        int rows_skipped
    }
    transactions {
        int id
        int source_id
        float amount
        datetime timestamp
        json features
        int dataset_label
        string row_hash
    }
    predictions {
        int id
        int transaction_id
        string model_version
        int predicted_label
        float risk_score
    }
    fraud_cases {
        int id
        int transaction_id
        string status
        string outcome
        text review_notes
        int reviewed_by
    }
    audit_logs {
        int id
        int user_id
        string action
        string subject
        json details
        string ip_address
    }
```

The imported record, the model's prediction and the analyst's outcome sit in three separate tables. A reviewer marking a case "Legitimate" never changes what the model predicted, so the two can be compared later. `dataset_label` keeps the Kaggle `Class` value for evaluation only, and the app never uses it to make decisions.

## The model

- **Data:** the Kaggle credit-card dataset, with 284,807 transactions and 492 frauds. 1,081 duplicate rows are dropped.
- **Split:** 60% train, 20% validation and 20% test, stratified so each part keeps the 0.17% fraud rate.
- **Pipeline:** `StandardScaler`, then `SMOTE`, then the classifier. SMOTE runs only while fitting, so validation and test rows are never resampled.
- **Candidates:** Logistic Regression, Decision Tree, Random Forest and a linear SVM.
- **Selection:** the highest PR-AUC on the validation set picks the model. Its alert threshold is the one that gives the best F1 on the validation set. The test set is scored once, at the end.
- **Result:** Random Forest, with test PR-AUC 0.788, precision 0.885 and recall 0.726 at threshold 0.815.

## Security

| Risk | Protection | Where |
|---|---|---|
| Stolen or guessed passwords | Passwords are stored as salted scrypt hashes and never in plain text. Login errors read the same for unknown emails and wrong passwords. | `models.User`, `routes/auth.py` |
| Password guessing | After 5 failed logins for one email from one IP address within 15 minutes, logins are refused with HTTP 429. The count resets after a successful login. | `services/audit.py`, `routes/auth.py` |
| Unauthorised access | Every page and API route needs a login. Importing and the audit log need the admin role. | `routes/auth.py` (`role_required`), `routes/pages.py` |
| Cross-site request forgery | Session cookies are `SameSite=Lax` and `HttpOnly`. API writes whose `Origin` header names another site are rejected. | `config.py`, `app.py` (`reject_cross_site_writes`) |
| Cross-site scripting | Pages insert user text with `textContent` only. A strict Content-Security-Policy allows only the app's own scripts and styles, with no inline code, and a test checks every page for inline code. | `static/js/common.js`, `app.py` |
| Clickjacking, MIME sniffing, caching | `X-Frame-Options: DENY`, `frame-ancestors 'none'`, `X-Content-Type-Options: nosniff`, and `Cache-Control: no-store` on API responses. | `app.py` (`security_headers`) |
| Open redirect after login | `?next=` only follows paths on the same site. | `routes/pages.py` (`_safe_next`) |
| Bad uploads | Files over 200 MB are refused. Files must be CSV with the expected columns, and invalid rows are skipped and counted. | `config.py`, `services/ingestion.py` |
| Untraceable changes | Logins, failed logins, imports, case changes and new users are written to `audit_logs`, in the same transaction as the change. | `services/audit.py` |
| Leaked secrets or data | `SECRET_KEY` comes from the environment. The dataset, database and trained model are git-ignored. | `config.py`, `.gitignore` |

The app never asks for or stores bank passwords, PINs, CVVs, OTPs or access tokens. A real bank integration would go through an authorised provider's API and consent flow, as section 10 of the project spec describes.

### Known limits

- The login limit counts attempts in the database, keyed by email and IP address. Someone using many IP addresses could still guess slowly. A production system would add a per-account lock and multi-factor login.
- `flask run` serves plain HTTP. In production, put the app behind HTTPS and set `SESSION_COOKIE_SECURE=1`.
- SQLite suits one server. The spec's planned move to PostgreSQL only needs `DATABASE_URL` changed, plus one SQLite-specific `strftime` call in `routes/dashboard.py`.
- A score above the threshold opens a case for a person to review. It never blocks a payment.
