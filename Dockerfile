# SecureBank production image: gunicorn serving the Flask app on port 8000.
# Train the models first (python -m ml.train ...): the image does not train them,
# and docker-compose.yml mounts ml/artifacts so retrained models persist.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Run as an ordinary user, who can write only the database and its backups (/data),
# Flask's instance folder and the models (so `flask retrain` works).
RUN useradd --create-home --uid 1000 securebank \
    && mkdir -p /data/backups /app/instance \
    && chown -R securebank /data /app/instance /app/ml/artifacts
USER securebank

ENV SECUREBANK_ENV=production \
    FLASK_APP=app \
    DATABASE_URL=sqlite:////data/securebank.db \
    BACKUP_DIR=/data/backups

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=4)"

# init-db is safe to run on every start: it only adds missing tables and columns.
# A long timeout, because a large CSV import runs inside the request.
CMD ["sh", "-c", "flask init-db && exec gunicorn --bind 0.0.0.0:8000 --workers 2 --timeout 300 --access-logfile - 'app:create_app()'"]
