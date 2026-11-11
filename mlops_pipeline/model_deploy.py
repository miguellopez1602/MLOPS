
"""API de predicción de incumplimiento de créditos."""

import json
from functools import lru_cache

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

try:
    from .ft_engineering import create_features, ROOT
except ImportError:
    from ft_engineering import create_features, ROOT


MODEL_PATH = ROOT / "artifacts" / "modelo_creditos.joblib"
METRICS_PATH = ROOT / "artifacts" / "metricas_modelo.json"

app = FastAPI(
    title="Predicción de Riesgo Crediticio",
    description="API de Machine Learning para predecir incumplimientos",
    version="1.0.0"
)


class BatchRequest(BaseModel):
    registros: list[dict] = Field(min_length=1, max_length=1000)


@lru_cache(maxsize=1)
def cargar_modelo():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "No existe el modelo. Ejecuta primero el entrenamiento."
        )
    return joblib.load(MODEL_PATH)


def obtener_umbral():
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            "No existe metricas_modelo.json. Ejecuta el entrenamiento."
        )
    configuracion = json.loads(
        METRICS_PATH.read_text(encoding="utf-8")
    )
    umbral = float(configuracion["umbral"])
    if not 0 <= umbral <= 1:
        raise ValueError("El umbral debe estar entre 0 y 1")
    return umbral


@app.get("/")
def inicio():
    return {
        "mensaje": "API de prediccion de riesgo crediticio",
        "documentacion": "/docs"
    }


@app.get("/health")
def estado():
    disponible = MODEL_PATH.exists() and METRICS_PATH.exists()
    return {
        "estado": "listo" if disponible else "faltan_artefactos",
        "modelo_disponible": MODEL_PATH.exists(),
        "configuracion_disponible": METRICS_PATH.exists()
    }


@app.post("/predict")
def predecir(payload: BatchRequest):
    try:
        modelo = cargar_modelo()
        umbral = obtener_umbral()

        datos = pd.DataFrame(payload.registros)
        caracteristicas = create_features(datos)

        probabilidades = modelo.predict_proba(caracteristicas)[:, 1]

        resultados = []
        for prob in probabilidades:
            prob = float(prob)
            clase = int(prob >= umbral)

            resultados.append({
                "probabilidad_incumplimiento": round(prob, 5),
                "clase_predicha": clase,
                "clasificacion": (
                    "Riesgo de incumplimiento"
                    if clase == 1
                    else "Sin alerta de incumplimiento"
                )
            })

        return {
            "modelo": "Modelo de riesgo crediticio",
            "umbral_utilizado": umbral,
            "cantidad": len(resultados),
            "predicciones": resultados
        }

    except FileNotFoundError as error:
        raise HTTPException(
            status_code=503, detail=str(error)
        ) from error

    except (ValueError, TypeError, KeyError) as error:
        raise HTTPException(
            status_code=422, detail=str(error)
        ) from error

