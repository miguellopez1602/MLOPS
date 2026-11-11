"""Pruebas de las cuatro vistas del monitor y sus filtros."""
import io
import json
import sys
from datetime import date
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest
from mlops_pipeline import model_monitoring as monitor


def credits():
    return pd.DataFrame([dict(fecha_prestamo='2025-03-15' if i<40 else '2025-04-15',
        capital_prestado=2000000+i*10000, plazo_meses=12, edad_cliente=30+i%10,
        salario_cliente=3000000+i*1000, cuota_pactada=200000,
        tipo_credito=4 if i<40 else 9, tipo_laboral='Empleado' if i<40 else 'Independiente',
        tendencia_ingresos='Estable', Pago_atiempo=int(i%5!=0)) for i in range(80)])


def test_dashboard_statistics():
    raw = credits()
    report = monitor._report_for_dashboard(raw.iloc[:40], raw.iloc[40:])
    assert len(report) == 19
    assert report.loc[report.variable=='tipo_credito','estado_magnitud'].iloc[0] == 'Rojo'
    assert monitor._js_from_probabilities([1,0],[0,1]) == pytest.approx(1)
    assert monitor._js_from_probabilities([1,1],[1,1]) == pytest.approx(0)
    assert np.isnan(monitor._js_from_probabilities([0,0],[1,1]))
    dates = monitor._parse_credit_dates(pd.Series([45658, '2025-04-01', None]))
    assert dates.iloc[1] == pd.Timestamp('2025-04-01')
    assert pd.isna(dates.iloc[2])
    assert monitor._data_by_period(raw,'Mensual').registros.sum() == 80
    assert monitor._data_by_period(raw,'Trimestral').registros.sum() == 80
    assert monitor._data_by_period(raw.drop(columns='fecha_prestamo'),'Mensual').empty
    assert monitor._data_by_period(raw.drop(columns='Pago_atiempo'),'Mensual').riesgo_real_pct.isna().all()
    assert monitor._format_percent(.25) == '25.0 %'
    assert monitor._format_percent(None) == '—'


@pytest.mark.parametrize('external, category, minimum, health_ok', [
    (False,False,30,True),(True,True,30,False),(False,True,500,False)])
def test_full_dashboard(monkeypatch,tmp_path,external,category,minimum,health_ok):
    raw = credits()
    folder = tmp_path/'artifacts'
    folder.mkdir()
    (folder/'metricas_modelo.json').write_text(json.dumps({'modelo':'Random_forest_ajustado',
        'umbral':.375,'evaluacion_optimizada':{'precision_riesgo':.15,'recall_riesgo':.24,
        'f2_riesgo':.21,'accuracy':.90}}))
    (folder/'historial_monitoreo.csv').write_text('fecha_utc,alertas,alertas_rojas\n2025-04-01,3,1\n')
    st = MagicMock()
    st.sidebar.date_input.return_value = date(2025,4,1)
    st.sidebar.radio.return_value = 'Trimestral' if category else 'Mensual'
    st.sidebar.number_input.return_value = minimum
    st.sidebar.file_uploader.return_value = io.StringIO(raw.iloc[40:].drop(columns='Pago_atiempo').to_csv(index=False)) if external else None
    st.columns.side_effect = lambda n: [MagicMock() for _ in range(n if isinstance(n,int) else len(n))]
    st.tabs.side_effect = lambda names: [MagicMock() for _ in names]
    st.selectbox.side_effect = lambda label, vals, **kw: ('Categóricas' if category else 'Numéricas') if label=='Filtrar tipo' else ('tipo_credito' if category else 'capital_prestado')
    st.checkbox.return_value = category
    st.text_input.return_value = 'http://127.0.0.1:8002/health'
    st.button.return_value = True
    import urllib.request
    if health_ok:
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'{"estado":"listo"}'
        monkeypatch.setattr(urllib.request,'urlopen',lambda *a,**kw: response)
    else:
        def unavailable(*a,**kw):
            raise OSError('API no disponible')
        monkeypatch.setattr(urllib.request,'urlopen',unavailable)
    monkeypatch.setitem(sys.modules,'streamlit',st)
    monkeypatch.setattr(monitor,'ROOT',tmp_path)
    monkeypatch.setattr(monitor,'load_data',lambda:raw)
    monitor.main_streamlit()
    assert st.tabs.call_count == 1
    assert st.pyplot.call_count == 2
    assert st.dataframe.call_count >= 2
    assert st.download_button.called
    assert st.bar_chart.called
    if health_ok:
        assert st.json.called
    if external:
        assert (folder/'monitoreo_drift.csv').is_file()


def test_dashboard_empty_reference(monkeypatch):
    raw = credits()
    st = MagicMock()
    st.sidebar.date_input.return_value = date(2025,3,15)
    st.sidebar.file_uploader.return_value = None
    monkeypatch.setitem(sys.modules,'streamlit',st)
    monkeypatch.setattr(monitor,'load_data',lambda:raw)
    monitor.main_streamlit()
    assert st.warning.called and not st.tabs.called
