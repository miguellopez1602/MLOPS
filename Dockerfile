# Reproducible: construye modelo y manifiesto durante docker build desde la base pública.
FROM python:3.11-slim AS training
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements-model.txt .
RUN pip install --no-cache-dir -r requirements-model.txt
COPY Base_de_datos.csv ./Base_de_datos.csv
COPY mlops_pipeline/ ./mlops_pipeline/
RUN python -m mlops_pipeline.model_training_evaluation

FROM python:3.11-slim AS api
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements-model.txt requirements-runtime.txt ./
RUN pip install --no-cache-dir -r requirements-runtime.txt && useradd --create-home apiuser
COPY mlops_pipeline/ ./mlops_pipeline/
COPY --from=training /app/artifacts/modelo_creditos.joblib /app/artifacts/modelo_creditos.joblib
COPY --from=training /app/artifacts/metricas_modelo.json /app/artifacts/metricas_modelo.json
USER apiuser
EXPOSE 8000
CMD ["uvicorn", "mlops_pipeline.model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]
