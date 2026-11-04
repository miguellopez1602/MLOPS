
"""Entrenamiento, optimización y comparación de modelos de créditos."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    fbeta_score, average_precision_score, confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

try:
    from .ft_engineering import build_preprocessor, prepare_training_data, ROOT
except ImportError:
    from ft_engineering import build_preprocessor, prepare_training_data, ROOT


ARTIFACTS = ROOT / "artifacts"


def build_model(classifier):
    return Pipeline([
        ("preprocesamiento", build_preprocessor()),
        ("modelo", clone(classifier))
    ])


def evaluar(y_real, probabilidades, umbral):
    predicciones = (probabilidades >= umbral).astype(int)

    return {
        "precision_riesgo": float(precision_score(y_real, predicciones, zero_division=0)),
        "recall_riesgo": float(recall_score(y_real, predicciones, zero_division=0)),
        "f1_riesgo": float(f1_score(y_real, predicciones, zero_division=0)),
        "f2_riesgo": float(fbeta_score(y_real, predicciones, beta=2, zero_division=0)),
        "pr_auc": float(average_precision_score(y_real, probabilidades))
    }


def main():
    ARTIFACTS.mkdir(exist_ok=True)

    # Se conserva intacto el conjunto de prueba original.
    X_train, X_test, y_train, y_test = prepare_training_data()

    # Separación adicional para seleccionar modelos y umbrales.
    X_dev, X_val, y_dev, y_val = train_test_split(
        X_train, y_train,
        test_size=0.25,
        stratify=y_train,
        random_state=42
    )

    print("Entrenamiento:", len(X_dev))
    print("Validacion:", len(X_val))
    print("Prueba:", len(X_test))

    modelos = {
        "Regresion_logistica": LogisticRegression(
            max_iter=2000, class_weight="balanced",
            solver="liblinear", random_state=42
        ),
        "Regresion_logistica_ajustada": LogisticRegression(
            C=0.3, max_iter=2000, class_weight="balanced",
            solver="liblinear", random_state=42
        ),
        "Random_forest": RandomForestClassifier(
            n_estimators=120, min_samples_leaf=4,
            class_weight="balanced_subsample",
            n_jobs=-1, random_state=42
        ),
        "Random_forest_ajustado": RandomForestClassifier(
            n_estimators=180, max_depth=12,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            n_jobs=-1, random_state=42
        )
    }

    resultados = []

    for nombre, clasificador in modelos.items():
        modelo = build_model(clasificador)
        modelo.fit(X_dev, y_dev)

        probabilidades = modelo.predict_proba(X_val)[:, 1]

        for umbral in np.arange(0.05, 0.751, 0.025):
            metricas = evaluar(y_val, probabilidades, umbral)

            resultados.append({
                "modelo": nombre,
                "umbral": round(float(umbral), 3),
                **metricas
            })

    tabla = pd.DataFrame(resultados)
    tabla.to_csv(ARTIFACTS / "busqueda_umbrales.csv", index=False)

    # Priorizamos recall, pero exigimos una precisión mínima.
    precision_minima = max(0.10, 2 * float(y_val.mean()))
    elegibles = tabla[tabla["precision_riesgo"] >= precision_minima]

    if elegibles.empty:
        elegibles = tabla
        print("Advertencia: ningun candidato alcanzo la precision minima.")

    mejor = elegibles.sort_values(
        ["f2_riesgo", "f1_riesgo", "pr_auc"],
        ascending=False
    ).iloc[0]

    nombre_mejor = str(mejor["modelo"])
    umbral_mejor = float(mejor["umbral"])

    print("\nMODELO SELECCIONADO EN VALIDACION:", nombre_mejor)
    print("Umbral seleccionado:", umbral_mejor)
    print("F2 validacion:", round(float(mejor["f2_riesgo"]), 4))

    # Mejor configuración por modelo, utilizando validación.
    comparacion = (
        tabla.sort_values("f2_riesgo", ascending=False)
        .drop_duplicates("modelo")
    )
    comparacion.to_csv(
        ARTIFACTS / "comparacion_modelos.csv", index=False
    )

    # Gráfica comparativa de validación.
    columnas = [
        "precision_riesgo", "recall_riesgo",
        "f1_riesgo", "f2_riesgo", "pr_auc"
    ]

    comparacion.set_index("modelo")[columnas].plot(
        kind="bar", figsize=(12, 6), rot=15
    )

    plt.title("Comparacion de modelos - Validacion")
    plt.ylabel("Puntaje")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "comparacion_modelos.png", dpi=180)
    plt.close()

    # Entrenar el modelo seleccionado con todo el train.
    modelo_final = build_model(modelos[nombre_mejor])
    modelo_final.fit(X_train, y_train)

    # El test se consulta solo al final.
    probabilidades_test = modelo_final.predict_proba(X_test)[:, 1]

    antes = evaluar(y_test, probabilidades_test, 0.50)
    despues = evaluar(y_test, probabilidades_test, umbral_mejor)

    print("\nRESULTADO EN PRUEBA - UMBRAL 0.50")
    print(antes)

    print("\nRESULTADO EN PRUEBA - UMBRAL OPTIMIZADO")
    print(despues)

    predicciones = (probabilidades_test >= umbral_mejor).astype(int)
    print("\nMATRIZ DE CONFUSION FINAL:")
    print(confusion_matrix(y_test, predicciones))

    # Gráfica antes y después del ajuste de umbral.
    comparacion_final = pd.DataFrame(
        [antes, despues],
        index=["Umbral 0.50", "Umbral optimizado"]
    )

    comparacion_final[
        ["precision_riesgo", "recall_riesgo", "f1_riesgo", "f2_riesgo"]
    ].plot(kind="bar", figsize=(9, 5), rot=0)

    plt.title("Modelo seleccionado: antes vs despues")
    plt.ylabel("Puntaje")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(ARTIFACTS / "comparacion_antes_despues.png", dpi=180)
    plt.close()

    joblib.dump(modelo_final, ARTIFACTS / "modelo_creditos.joblib")

    informacion = {
        "modelo": nombre_mejor,
        "umbral": umbral_mejor,
        "precision_minima_validacion": precision_minima,
        "evaluacion_original": antes,
        "evaluacion_optimizada": despues
    }

    (ARTIFACTS / "metricas_modelo.json").write_text(
        json.dumps(informacion, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("\nModelo y graficas guardados correctamente.")


if __name__ == "__main__":
    main()