# Deploying SecureBank

This guide runs SecureBank on a Linux server with Docker, behind HTTPS. `flask run` is only for development: it serves plain HTTP, handles one request at a time, and shows no access log.

The setup has two containers:

```
Browser / payment system ──HTTPS──> Caddy (ports 80, 443) ──HTTP──> gunicorn + Flask app (port 8000, not published)
                                                                       │
                                                         /data volume: securebank.db, backups/
```

- **Caddy** gets a certificate from Let's Encrypt for your domain, renews it, redirects HTTP to HTTPS, and passes requests to the app with the visitor's address in `X-Forwarded-For`.
- **gunicorn** runs the Flask app with two worker processes and logs every request to the container's output.
- The **database** and its **backups** live in a Docker volume, so they survive rebuilding the image.
- The trained **models** are mounted from `ml/artifacts/` on the server, so a retrained model survives too.

## 1. Before you start

You need Docker with the Compose plugin, and for a real certificate a domain name whose DNS points at the server, with ports 80 and 443 open. For a demo on your own machine, `localhost` works and Caddy uses its own certificate, which the browser will warn about once.

Train the models on the server, or copy them there, before starting. The image does not train them:

```bash
python -m ml.train                     # writes ml/artifacts/creditcard/model.joblib
```

## 2. Settings

```bash
cp deploy/env.example .env
python -c "import secrets; print(secrets.token_hex(32))"   # run twice, once per key
```

Edit `.env`:

| Setting | What it is |
|---|---|
| `SECRET_KEY` | Signs login cookies. Changing it logs everyone out. |
| `CARD_ID_KEY` | Keys the hash that links a card's payments without storing its number. Set it once, before the first import: changing it later breaks the link between a card's old and new payments. |
| `SITE_ADDRESS` | Your domain, such as `fraud.example.com`, or `localhost`. |

`.env` is git-ignored. Never commit it.

The image sets `SECUREBANK_ENV=production`. In production mode the app:

- **refuses to start** if `SECRET_KEY` or `CARD_ID_KEY` is missing, shorter than 32 characters, or still the example value, so a server cannot run with development secrets by mistake;
- **sends cookies only over HTTPS** (`Secure`), and tells browsers to use HTTPS for a year (`Strict-Transport-Security`);
- **logs** each error with its traceback to the container output, while the visitor sees only "Something went wrong".

`docker-compose.yml` also sets `TRUST_PROXY=1`, so the login limit and the audit log see the visitor's address rather than Caddy's. Set it only when a proxy you control is the only way to reach the app; otherwise anyone could fake their address in a header.

Other optional settings, such as `DECLINE_SCORE`, `EXPORT_MAX_ROWS` and `REQUIRE_2FA_FOR_ADMINS`, are described in `config.py`.

## 3. Start it

```bash
docker compose up -d --build
docker compose exec app flask create-user admin@yourbank.example --role admin
```

The app creates or updates its tables on every start. Open `https://<SITE_ADDRESS>`, log in, and set up two-factor login on the Account page, which admins must do before anything else.

Check that it is running:

```bash
docker compose ps                      # app should show "healthy"
curl https://<SITE_ADDRESS>/api/health # {"database":"ok","status":"ok"}
docker compose logs -f app             # requests and errors
```

`/api/health` returns 503 when the database cannot be reached, so an uptime monitor or load balancer can use it.

Large files can also be imported from the command line, which avoids the browser waiting on a long upload. Put the file in `data/raw/` on the server (mounted read-only into the container):

```bash
docker compose exec app flask import-csv data/raw/fraudTest.csv
```

## 4. Backups

`flask backup-db` copies the database into `BACKUP_DIR` (`/data/backups` in the container) using SQLite's backup API, which takes a consistent copy even while the app is writing. It keeps the newest 14 copies and deletes older ones (`--keep` changes that).

Run it every night from the server's crontab (`crontab -e`):

```
30 2 * * * cd /opt/secure-bank && docker compose exec -T app flask backup-db >> /var/log/securebank-backup.log 2>&1
```

The backups are in the same Docker volume as the database, so they protect against mistakes and corruption but not against losing the server. Copy them off the server too, for example:

```bash
docker compose cp app:/data/backups ./backups
```

To restore, stop the app, copy a backup over the database and start it again:

```bash
docker compose stop app
docker compose cp ./backups/securebank-20261015T023000Z.db app:/data/securebank.db
docker compose start app
```

## 5. Updating and retraining

```bash
git pull
docker compose up -d --build           # rebuilds the image; the database volume is kept
```

To retrain a model with the analysts' case decisions (see the README), run it inside the container and restart so every worker loads the new model:

```bash
docker compose exec app flask retrain --dataset sparkov --data data/raw/fraudTrain.csv --test-data data/raw/fraudTest.csv
docker compose restart app
```

The container runs as user ID 1000, which must be able to write `ml/artifacts/` on the server for this (`sudo chown -R 1000 ml/artifacts`).

## Running without Docker

The same settings work on a plain server:

```bash
export SECUREBANK_ENV=production SECRET_KEY=... CARD_ID_KEY=... TRUST_PROXY=1
flask --app app init-db
gunicorn --bind 127.0.0.1:8000 --workers 2 --timeout 300 'app:create_app()'
```

Put nginx or Caddy in front for HTTPS, and schedule `flask --app app backup-db` with cron.

## Known limits

- SQLite allows one writer at a time, which suits one server and a small team. More traffic or several servers need PostgreSQL (change `DATABASE_URL`, and the one SQLite-specific `strftime` call in `routes/dashboard.py`).
- Imports from the web page run inside the request, hence gunicorn's 300-second timeout. Very large files are better imported with `flask import-csv`.
- Logs go to the container output only. A real deployment would ship them to a log service and alert on errors.
