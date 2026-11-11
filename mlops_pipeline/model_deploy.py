"""FastAPI para inferencia individual y por lotes, con contrato de datos validado."""
import hashlib
import json
from functools import lru_cache
from typing import Literal
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
try:
    from .ft_engineering import create_features, ROOT
except ImportError:
    from ft_engineering import create_features, ROOT

MODEL_PATH = ROOT/'artifacts'/'modelo_creditos.joblib'
METRICS_PATH = ROOT/'artifacts'/'metricas_modelo.json'
app = FastAPI(title='Predicción de Riesgo Crediticio', version='2.0.0',
              description='Demostración educativa: no usar para negar créditos reales.')


class CreditRecord(BaseModel):
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)
    capital_prestado: float = Field(gt=0, le=100_000_000)
    plazo_meses: int = Field(ge=1, le=360)
    edad_cliente: int = Field(ge=18, le=100)
    salario_cliente: float = Field(gt=0, le=100_000_000)
    cuota_pactada: float = Field(ge=0, le=100_000_000)
    tipo_credito: int = Field(ge=0)
    tipo_laboral: Literal['Empleado', 'Independiente']
    total_otros_prestamos: float | None = Field(default=None, ge=0)
    puntaje_datacredito: float | None = Field(default=None, ge=1, le=1000)
    cant_creditosvigentes: int | None = Field(default=None, ge=0)
    huella_consulta: int | None = Field(default=None, ge=0)
    creditos_sectorFinanciero: int | None = Field(default=None, ge=0)
    creditos_sectorCooperativo: int | None = Field(default=None, ge=0)
    creditos_sectorReal: int | None = Field(default=None, ge=0)
    promedio_ingresos_datacredito: float | None = Field(default=None, ge=0)
    tendencia_ingresos: Literal['Creciente','Estable','Decreciente'] | None = None
    fecha_prestamo: str | float | None = None

    @model_validator(mode='after')
    def validar_cuota(self):
        if self.cuota_pactada > self.capital_prestado * 2:
            raise ValueError('Cuota pactada desproporcionada respecto al capital')
        return self


class BatchRequest(BaseModel):
    registros: list[CreditRecord] = Field(min_length=1, max_length=1000)


@lru_cache(maxsize=1)
def cargar_modelo():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError('No se encontró el modelo. Ejecute entrenamiento o construya Docker.')
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def obtener_configuracion():
    if not METRICS_PATH.is_file():
        raise FileNotFoundError('No se encontró metricas_modelo.json.')
    cfg = json.loads(METRICS_PATH.read_text(encoding='utf-8'))
    threshold = float(cfg['umbral'])
    if not 0 <= threshold <= 1:
        raise ValueError('El umbral no pertenece a [0,1]')
    # El manifiesto detecta mezclas accidentales de modelo y metadatos.
    expected = cfg.get('sha256_modelo')
    if expected and MODEL_PATH.exists():
        observed = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
        if observed != expected:
            raise ValueError('El hash del modelo no coincide con el manifiesto de métricas')
    return cfg


@app.get('/')
def inicio():
    return {'mensaje': 'API de riesgo crediticio (solo educativo)', 'documentacion': '/docs'}


@app.get('/health')
def estado():
    try:
        cfg = obtener_configuracion()
        model = cargar_modelo()
        if not hasattr(model, 'predict_proba'):
            raise TypeError('El objeto no es un clasificador probabilístico')
        return {'estado': 'listo', 'modelo_disponible': True,
                'configuracion_disponible': True, 'modelo': cfg.get('modelo'),
                'umbral': float(cfg['umbral'])}
    except (FileNotFoundError, ValueError, TypeError, OSError, ImportError, KeyError) as exc:
        raise HTTPException(status_code=503, detail={
            'estado': 'no_listo', 'modelo_disponible': MODEL_PATH.exists(),
            'configuracion_disponible': METRICS_PATH.exists(), 'detalle': str(exc)
        }) from exc


@app.post('/predict')
def predecir(payload: BatchRequest):
    try:
        modelo = cargar_modelo()
        config = obtener_configuracion()
        threshold = float(config['umbral'])
        datos = pd.DataFrame([r.model_dump(exclude_none=True) for r in payload.registros])
        proba = modelo.predict_proba(create_features(datos))[:, 1]
        results = []
        for p in proba:
            clase = int(float(p) >= threshold)
            results.append({'probabilidad_incumplimiento': round(float(p), 5),
                            'clase_predicha': clase,
                            'clasificacion': 'Riesgo de incumplimiento' if clase else 'Sin alerta de incumplimiento'})
        return {'modelo': config.get('modelo','Crédito'), 'umbral_utilizado': threshold,
                'cantidad': len(results), 'predicciones': results,
                'advertencia': 'Resultados experimentales; no usar para decisiones crediticias.'}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, TypeError, KeyError, OSError, ImportError) as exc:
        raise HTTPException(status_code=503, detail=f'Modelo o configuración no disponible: {exc}') from exc
