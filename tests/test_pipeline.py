"""Pruebas automatizadas de features, entrenamiento, API y drift (sin datos sensibles)."""
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from mlops_pipeline.ft_engineering import (
    create_features, FEATURE_COLS, LEAKAGE_CANDIDATES,
    prepare_training_data, build_preprocessor,
)
from mlops_pipeline.model_training_evaluation import summarize_classification
from mlops_pipeline.model_monitoring import calculate_drift, population_stability_index
from mlops_pipeline import model_deploy


def _credit(salary=4_000_000):
    return {'capital_prestado': 2_000_000, 'plazo_meses': 12, 'edad_cliente': 30,
            'salario_cliente': salary, 'cuota_pactada': 400_000,
            'tipo_credito': 4, 'tipo_laboral': 'Empleado',
            'tendencia_ingresos': 'Estable'}


def test_feature_columns_ratios():
    features = create_features(pd.DataFrame([_credit()]))
    assert list(features) == FEATURE_COLS
    assert features['ratio_cuota_salario'].iloc[0] == .1
    assert features['ratio_capital_salario'].iloc[0] == .5


def test_no_target_or_leakage():
    d = _credit(); d.update({'Pago_atiempo': 0, 'saldo_mora': 500000, 'puntaje': 95})
    features = create_features(pd.DataFrame([d]))
    assert not any(x in features for x in LEAKAGE_CANDIDATES + ['Pago_atiempo'])


def test_invalid_age_and_salary_to_missing():
    f = create_features(pd.DataFrame([{'edad_cliente':123, 'salario_cliente': 22_000_000_000}]))
    assert pd.isna(f['edad_cliente'].iloc[0])
    assert pd.isna(f['salario_cliente'].iloc[0])


def test_date_serial_to_month():
    f = create_features(pd.DataFrame([{'fecha_prestamo': 45664.611528}]))
    assert int(f['mes_prestamo'].iloc[0]) in range(1, 13)


def test_split_stratified_and_target_converted():
    df = pd.DataFrame([_credit() | {'Pago_atiempo': i % 5 != 0} for i in range(100)])
    X_tr, X_te, y_tr, y_te = prepare_training_data(df, random_state=42)
    assert len(X_tr) == 80 and len(X_te) == 20
    assert set(y_tr.unique()) == {0, 1} and sum(y_te) == 4
    assert 'Pago_atiempo' not in X_tr.columns


def test_preprocessor_does_not_learn_before_fit():
    prep = build_preprocessor()
    assert not hasattr(prep, 'transformers_')


def test_summary_metrics_and_confusion():
    y = [0, 0, 1, 1]
    p = [.01, .12, .89, .78]
    m = summarize_classification(y, p, .5)
    assert m['accuracy'] == 1.0
    assert m['tp'] == 2 and m['fn'] == 0 and m['fp'] == 0
    assert m['roc_auc'] == 1.0
    assert m['pr_auc'] == 1.0


def test_psi_no_drift():
    x = np.linspace(0, 1, 100)
    assert abs(population_stability_index(x, x)) < .000001


def test_psi_constant_reference_change():
    assert population_stability_index([1]*50, [2]*50) > .25


def test_monitor_drift_without_target():
    ref = pd.DataFrame([_credit(salary=2_000_000 + i*1000) for i in range(60)])
    curr = pd.DataFrame([_credit(salary=25_000_000 + i*1000) for i in range(60)])
    rep = calculate_drift(ref, curr)
    row = rep.loc[rep.variable=='salario_cliente'].iloc[0]
    assert bool(row.alerta) is True
    assert row.severidad == 'Rojo'
    assert 'n_actual' in rep


client = TestClient(model_deploy.app)


def test_api_rejects_empty():
    resp = client.post('/predict', json={'registros': [{}]})
    assert resp.status_code == 422


def test_api_rejects_negative_income():
    resp = client.post('/predict', json={'registros': [_credit(salary=-1)]})
    assert resp.status_code == 422


def test_api_rejects_leakage_fields():
    invalid = _credit() | {'puntaje': 99}
    resp = client.post('/predict', json={'registros': [invalid]})
    assert resp.status_code == 422


def test_api_predict_batch(monkeypatch):
    class FakeModel:
        def predict_proba(self, frame):
            return np.array([[.9, .1], [.7, .3]])[:len(frame)]
    monkeypatch.setattr(model_deploy, 'cargar_modelo', lambda: FakeModel())
    monkeypatch.setattr(model_deploy, 'obtener_configuracion', lambda: {'modelo':'Test', 'umbral': .2})
    resp = client.post('/predict', json={'registros': [_credit(), _credit(2_000_000)]})
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data['cantidad'] == 2 and data['umbral_utilizado'] == .2
    assert [x['clase_predicha'] for x in data['predicciones']] == [0, 1]


def test_health_ready(monkeypatch):
    monkeypatch.setattr(model_deploy, 'cargar_modelo', lambda: type('TestModel', (), {'predict_proba': lambda x: None})())
    monkeypatch.setattr(model_deploy, 'obtener_configuracion', lambda: {'modelo': 'Test', 'umbral': .2})
    assert client.get('/health').json()['estado'] == 'listo'


def test_training_generates_consistent_artifacts(monkeypatch, tmp_path):
    """Integración: buscar umbral, entrenar, generar gráficos y manifiesto juntos."""
    from mlops_pipeline import model_training_evaluation as trainer
    from mlops_pipeline.ft_engineering import load_data
    from sklearn.model_selection import train_test_split
    data = load_data()
    small = pd.concat([
        data.loc[data.Pago_atiempo == 0].sample(n=60, random_state=42),
        data.loc[data.Pago_atiempo == 1].sample(n=440, random_state=42),
    ]).sample(frac=1, random_state=42)
    x_tr, x_te, y_tr, y_te = prepare_training_data(small)
    monkeypatch.setattr(trainer, 'prepare_training_data', lambda: (x_tr, x_te, y_tr, y_te))
    monkeypatch.setattr(trainer, 'ARTIFACTS', tmp_path)
    report = trainer.main()
    assert report['modelo'] in trainer.candidatos()
    assert 0 <= report['umbral'] <= 1
    assert report['test'] == 100
    assert (tmp_path / 'modelo_creditos.joblib').is_file()
    assert (tmp_path / 'metricas_modelo.json').is_file()
    assert (tmp_path / 'matriz_confusion.png').is_file()
    assert (tmp_path / 'curvas_roc_pr.png').is_file()
    assert (tmp_path / 'busqueda_umbrales.csv').is_file()
    import hashlib, json
    metadata = json.loads((tmp_path / 'metricas_modelo.json').read_text())
    actual_hash = hashlib.sha256((tmp_path/'modelo_creditos.joblib').read_bytes()).hexdigest()
    assert metadata['sha256_modelo'] == actual_hash


def test_monitor_cli_accepts_unlabeled_csv(monkeypatch, tmp_path):
    from mlops_pipeline import model_monitoring as monitor
    import sys
    src = pd.DataFrame([_credit() for _ in range(25)])
    current = pd.DataFrame([_credit(2_000_000) for _ in range(25)])
    fname = tmp_path/'new.csv'
    current.to_csv(fname, index=False)
    monkeypatch.setattr(monitor, 'demo_datasets', lambda path: (src, current, False))
    monkeypatch.setattr(monitor, 'ROOT', tmp_path)
    monkeypatch.setattr(sys, 'argv', ['model_monitoring.py','--actual',str(fname)])
    monitor.main_cli()
    assert (tmp_path/'artifacts'/'monitoreo_drift.csv').is_file()
    assert (tmp_path/'artifacts'/'historial_monitoreo.csv').is_file()


def test_monitor_streamlit_dashboard_demo(monkeypatch, tmp_path):
    """Prueba sin servidor HTTP del flujo completo de visualización Streamlit."""
    from mlops_pipeline import model_monitoring as monitor
    from types import SimpleNamespace
    import sys
    raw = pd.DataFrame([_credit(salary=2_000_000 + i * 500) for i in range(40)])
    called=[]
    def log(name):
        return lambda *args, **kwargs: called.append(name)
    col = SimpleNamespace(metric=log('metric'))
    fake = SimpleNamespace(set_page_config=log('config'), title=log('title'),
        caption=log('caption'), file_uploader=lambda *a, **kw: None,
        warning=log('warning'), error=log('error'), success=log('success'),
        columns=lambda n: [col]*n, subheader=log('subheader'),
        dataframe=log('dataframe'), bar_chart=log('bar_chart'),
        selectbox=lambda label, vals: vals[0], pyplot=log('plot'),
        info=log('info'), button=lambda *a, **kw: False,
        line_chart=log('line_chart'))
    monkeypatch.setitem(sys.modules, 'streamlit', fake)
    monkeypatch.setattr(monitor, 'load_data', lambda: raw)
    monkeypatch.setattr(monitor, 'ROOT', tmp_path)
    monitor.main_streamlit()
    assert {'title','dataframe','plot','warning'}.issubset(set(called))


def test_health_returns_503_if_model_missing(monkeypatch):
    def broken():
        raise FileNotFoundError('Modelo ausente')
    monkeypatch.setattr(model_deploy, 'obtener_configuracion', broken)
    assert client.get('/health').status_code == 503


def test_model_hash_detects_mismatched_manifest(monkeypatch, tmp_path):
    import json
    file_model = tmp_path/'model.joblib'
    file_model.write_bytes(b'datos falsos')
    conf = tmp_path/'metricas.json'
    conf.write_text(json.dumps({'umbral': .2, 'sha256_modelo': 'f'*64}))
    monkeypatch.setattr(model_deploy, 'MODEL_PATH', file_model)
    monkeypatch.setattr(model_deploy, 'METRICS_PATH', conf)
    model_deploy.obtener_configuracion.cache_clear()
    with pytest.raises(ValueError, match='hash'):
        model_deploy.obtener_configuracion()
    model_deploy.obtener_configuracion.cache_clear()
