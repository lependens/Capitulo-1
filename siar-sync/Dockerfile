FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN CERTIFI_CA="$(python -c 'import certifi; print(certifi.where())')" \
    && { cat "$CERTIFI_CA"; printf '\n'; cat /app/certs/siar_extra_chain.pem; } > /app/certs/siar_bundle.pem \
    && chmod 0644 /app/certs/siar_bundle.pem
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
