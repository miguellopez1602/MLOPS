# MLOPS — Predicción de Riesgo Crediticio

## Requisitos
- Python 3.11
- Git
- Docker (opcional)

## 1. Instalación

`py -3.11 -m venv .venv`

`.\.venv\Scripts\python.exe -m pip install -r requirements.txt`

## 2. Entrenar el modelo

`.\.venv\Scripts\python.exe -m mlops_pipeline.model_training_evaluation`

## 3. Ejecutar la API

`.\.venv\Scripts\python.exe -m uvicorn mlops_pipeline.model_deploy:app --reload`

Documentación: http://localhost:8000/docs

## 4. Ejecutar el monitoreo

`.\.venv\Scripts\python.exe -m streamlit run mlops_pipeline/model_monitoring.py --server.port 8105`

Panel: http://localhost:8105

## 5. Ejecutar las pruebas

`.\.venv\Scripts\python.exe -m pytest -q tests/`

## 6. Docker

`docker build -t mlops-corregido .`

`docker run --rm -d -p 8002:8000 mlops-corregido`

API: http://localhost:8002/docs

## 7. Integración continua

- **GitHub Actions:** análisis con SonarQube Cloud.
- **Jenkins:** pipeline `MLOPS-CI` para ejecutar pruebas automatizadas.

## Repositorio

https://github.com/miguellopez1602/MLOPS
