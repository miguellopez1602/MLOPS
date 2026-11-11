"""Entrena y selecciona clasificadores sin utilizar el conjunto de test para ajustar.

Reproducibilidad: versiones fijadas en requirements-model.txt; semilla 42;
artefactos y reporte JSON generados juntos desde la misma ejecución.
"""
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
    fbeta_score, average_precision_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
try:
    from .ft_engineering import build_preprocessor, prepare_training_data, ROOT, DATA_PATH
except ImportError:
    from ft_engineering import build_preprocessor, prepare_training_data, ROOT, DATA_PATH

ARTIFACTS = ROOT / 'artifacts'
RANDOM_STATE = 42


def build_model(classifier):
    return Pipeline([('preprocesamiento', build_preprocessor()), ('modelo', clone(classifier))])


def summarize_classification(y_real, probabilidades, umbral=0.5):
    """Resumen con métrica principal para riesgo=1; la clase positiva es incumplimiento."""
    pred = (np.asarray(probabilidades) >= umbral).astype(int)
    cm = confusion_matrix(y_real, pred, labels=[0, 1])
    return {
        'accuracy': float(accuracy_score(y_real, pred)),
        'precision_riesgo': float(precision_score(y_real, pred, zero_division=0)),
        'recall_riesgo': float(recall_score(y_real, pred, zero_division=0)),
        'f1_riesgo': float(f1_score(y_real, pred, zero_division=0)),
        'f2_riesgo': float(fbeta_score(y_real, pred, beta=2, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_real, probabilidades)),
        'pr_auc': float(average_precision_score(y_real, probabilidades)),
        'tn': int(cm[0, 0]), 'fp': int(cm[0, 1]),
        'fn': int(cm[1, 0]), 'tp': int(cm[1, 1]),
    }


def evaluar(y_real, probabilidades, umbral):
    """Alias para conservar compatibilidad con scripts anteriores."""
    return summarize_classification(y_real, probabilidades, umbral)


def candidatos():
    return {
        'Regresion_logistica': LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear', random_state=42),
        'Regresion_logistica_ajustada': LogisticRegression(C=0.3, max_iter=2000, class_weight='balanced', solver='liblinear', random_state=42),
        'Random_forest': RandomForestClassifier(n_estimators=120, min_samples_leaf=4, class_weight='balanced_subsample', n_jobs=-1, random_state=42),
        'Random_forest_ajustado': RandomForestClassifier(n_estimators=180, max_depth=12, min_samples_leaf=8, class_weight='balanced_subsample', n_jobs=-1, random_state=42),
    }


def plot_outputs(comparison, y_test, proba_test, threshold, before, after):
    names = ['precision_riesgo', 'recall_riesgo', 'f1_riesgo', 'f2_riesgo', 'pr_auc']
    ax = comparison.set_index('modelo')[names].plot(kind='bar', figsize=(12, 6), rot=15, width=.8)
    ax.set_title('Mejor configuración ELEGIBLE por modelo (validación, precisión mínima)')
    ax.set_ylabel('Puntaje'); ax.set_ylim(0, max(.55, min(1.0, comparison[names].max().max() + .13)))
    plt.tight_layout(); plt.savefig(ARTIFACTS/'comparacion_modelos.png', dpi=170); plt.close()

    change = pd.DataFrame([before, after], index=['Umbral 0.50', f'Umbral {threshold:.3f}'])
    ax = change[['precision_riesgo', 'recall_riesgo', 'f1_riesgo', 'f2_riesgo']].plot(kind='bar', figsize=(9, 5), rot=0)
    ax.set_title('Compromiso precisión/recall en prueba (mismo modelo)'); ax.set_ylim(0, 1)
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x()+p.get_width()/2, p.get_height()+.012),
                    ha='center', va='bottom', fontsize=7)
    plt.tight_layout(); plt.savefig(ARTIFACTS/'comparacion_antes_despues.png', dpi=170); plt.close()

    pred = (proba_test >= threshold).astype(int)
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_test, pred, labels=[0, 1],
        display_labels=['Sin incumplimiento', 'Incumplimiento'], ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f'Matriz de confusión en prueba (umbral {threshold:.3f})')
    plt.tight_layout(); plt.savefig(ARTIFACTS/'matriz_confusion.png', dpi=170); plt.close()

    fpr, tpr, _ = roc_curve(y_test, proba_test)
    prec, rec, _ = precision_recall_curve(y_test, proba_test)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
    axs[0].plot(fpr, tpr, label=f'ROC-AUC={after["roc_auc"]:.3f}'); axs[0].plot([0,1], [0,1], '--', color='gray')
    axs[0].set(xlabel='FPR', ylabel='Recall / TPR', title='Curva ROC'); axs[0].legend()
    axs[1].plot(rec, prec, label=f'PR-AUC={after["pr_auc"]:.3f}'); axs[1].axhline(np.mean(y_test), ls='--', color='gray', label='Prevalencia')
    axs[1].set(xlabel='Recall', ylabel='Precisión', title='Curva precisión-recall'); axs[1].legend()
    plt.tight_layout(); plt.savefig(ARTIFACTS/'curvas_roc_pr.png', dpi=170); plt.close()


def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = prepare_training_data()
    X_dev, X_val, y_dev, y_val = train_test_split(X_train, y_train, test_size=.25,
                                                   stratify=y_train, random_state=RANDOM_STATE)
    modelos = candidatos()
    resultados = []
    start = time.perf_counter()
    for name, classifier in modelos.items():
        model = build_model(classifier).fit(X_dev, y_dev)
        proba = model.predict_proba(X_val)[:, 1]
        for threshold in np.round(np.arange(.05, .751, .025), 3):
            metric = summarize_classification(y_val, proba, float(threshold))
            resultados.append({'modelo': name, 'umbral': float(threshold), **metric})
    table = pd.DataFrame(resultados)
    p_min = max(.10, 2 * float(y_val.mean()))
    table['cumple_precision'] = table['precision_riesgo'] >= p_min
    table.to_csv(ARTIFACTS/'busqueda_umbrales.csv', index=False)
    eligible = table.loc[table.cumple_precision].copy()
    fallback = eligible.empty
    if fallback:
        # No afirmar que se cumple el criterio de negocio si no se encontró candidato.
        eligible = table.copy()
        print('ADVERTENCIA: ningún candidato cumplió la precisión mínima.')
    ordered = eligible.sort_values(['f2_riesgo','f1_riesgo','pr_auc','modelo'], ascending=[False,False,False,True])
    best = ordered.iloc[0]
    name, threshold = str(best.modelo), float(best.umbral)
    comparison = (ordered.drop_duplicates('modelo').sort_values('modelo'))
    comparison.to_csv(ARTIFACTS/'comparacion_modelos.csv', index=False)
    # Prueba de estabilidad solo sobre train; no se usa el test para elegir el ganador.
    kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    cv_ap = cross_val_score(build_model(modelos[name]), X_train, y_train,
                             cv=kfold, scoring='average_precision', n_jobs=1)
    model = build_model(modelos[name]).fit(X_train, y_train)
    elapsed = time.perf_counter() - start
    t0 = time.perf_counter()
    prob_test = model.predict_proba(X_test)[:, 1]
    inference = time.perf_counter() - t0
    before = summarize_classification(y_test, prob_test, .5)
    after = summarize_classification(y_test, prob_test, threshold)
    plot_outputs(comparison, y_test, prob_test, threshold, before, after)
    model_path = ARTIFACTS/'modelo_creditos.joblib'
    joblib.dump(model, model_path)
    info = {
        'modelo': name, 'umbral': threshold, 'umbral_original': .5,
        'precision_minima_validacion': p_min, 'criterio_precision_cumplido': not fallback,
        'objetivo': '1 = incumplimiento', 'train': len(X_train), 'dev': len(X_dev),
        'validacion': len(X_val), 'test': len(X_test),
        'f2_validacion': float(best.f2_riesgo),
        'evaluacion_original': before, 'evaluacion_optimizada': after,
        'matriz_confusion': [[after['tn'],after['fp']],[after['fn'],after['tp']]],
        'consistencia_pr_auc_cv3_mean': float(cv_ap.mean()),
        'consistencia_pr_auc_cv3_std': float(cv_ap.std()),
        'pr_auc_cv3_fold': [float(x) for x in cv_ap],
        'segundos_entrenamiento_y_seleccion': round(elapsed, 3),
        'ms_inferencia_por_registro': round(inference / len(X_test) * 1000, 4),
        'python_version': platform.python_version(), 'sklearn_version': sklearn.__version__,
        'pandas_version': pd.__version__, 'numpy_version': np.__version__,
        'sha256_dataset': hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        'sha256_modelo': hashlib.sha256(model_path.read_bytes()).hexdigest(),
        'entrenado_utc': datetime.now(timezone.utc).isoformat(),
        'limitacion': 'Uso educativo; no utilizar para rechazar créditos reales.',
    }
    (ARTIFACTS/'metricas_modelo.json').write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding='utf-8')
    print('Modelo:', name, '| Umbral:', threshold, '| Validación F2:', round(float(best.f2_riesgo),4))
    print('CV PR-AUC:', round(cv_ap.mean(), 4), '+/-', round(cv_ap.std(), 4))
    print('TEST, umbral 0.50:', json.dumps(before, ensure_ascii=False))
    print('TEST, umbral óptimo:', json.dumps(after, ensure_ascii=False))
    print(classification_report(y_test, (prob_test >= threshold).astype(int), zero_division=0))
    print('Artefactos guardados en:', ARTIFACTS)
    return info


if __name__ == '__main__':
    main()
