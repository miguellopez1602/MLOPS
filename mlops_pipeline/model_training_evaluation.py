"""Entrenamiento supervisado y comparación de modelos"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline

try:
    from .ft_engineering import build_preprocessor, prepare_training_data, ROOT
except ImportError:
    from ft_engineering import build_preprocessor, prepare_training_data, ROOT

ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "modelo_creditos.joblib"


def summarize_classification(y_true, y_pred, y_score):
    """Métricas enfocadas en la clase 1 = riesgo de incumplimiento."""
    return {
        "precision_riesgo": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_riesgo": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_riesgo": float(f1_score(y_true, y_pred, zero_division=0)),
        "average_precision_pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }


def build_model(classifier):
    return Pipeline([("preprocesamiento", build_preprocessor()), ("modelo", clone(classifier))])


def main():
    X_train, X_test, y_train, y_test = prepare_training_data()
    print(f"Train {len(X_train)} / test {len(X_test)} | riesgo: {y_train.mean():.2%}")
    candidates = {
        "Baseline_mayoria": DummyClassifier(strategy="prior"),
        "Regresion_logistica": LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=42),
        "Random_forest": RandomForestClassifier(n_estimators=120, min_samples_leaf=4,
                                                  class_weight="balanced_subsample", n_jobs=-1, random_state=42),
    }
    results = []
    trained = {}
    for name, estimator in candidates.items():
        pipe = build_model(estimator)
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        y_pred = pipe.predict(X_test)
        metrics = summarize_classification(y_test, y_pred, y_prob)
        results.append({"modelo": name, **metrics})
        trained[name] = pipe
        print(f"\n{name}:", metrics)
        print("Matriz de confusión [real filas, predicción columnas]:\n", confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred, zero_division=0))

    summary = pd.DataFrame(results).sort_values("average_precision_pr_auc", ascending=False)
    ARTIFACTS.mkdir(exist_ok=True)
    summary.to_csv(ARTIFACTS / "comparacion_modelos.csv", index=False)
    best_name = summary.iloc[0]["modelo"]
    # Nunca seleccionamos el dummy como modelo productivo.
    if best_name == "Baseline_mayoria":
        best_name = summary.loc[summary["modelo"] != "Baseline_mayoria"].iloc[0]["modelo"]
    best_model = trained[best_name]
    joblib.dump(best_model, MODEL_PATH)
    (ARTIFACTS / "metricas_modelo.json").write_text(
        json.dumps({"modelo": best_name, "evaluacion_test": summary.to_dict(orient="records"),
                    "interpretacion": "1 = riesgo de NO pagar a tiempo"}, indent=2, ensure_ascii=False),
        encoding="utf-8")
    print(f"\nSeleccionado: {best_name}. Guardado en {MODEL_PATH}")


if __name__ == "__main__":
    main()
