"""Data drift de características (sin etiqueta obligatoria). Demo ≠ producción."""
import argparse
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
try:
    from .ft_engineering import load_data, create_features, ROOT, NUMERIC_COLS, CATEGORICAL_COLS
except ImportError:
    from ft_engineering import load_data, create_features, ROOT, NUMERIC_COLS, CATEGORICAL_COLS

MONITORED_NUMERIC = NUMERIC_COLS + ['mes_prestamo','ratio_cuota_salario','ratio_capital_salario']


def population_stability_index(reference, actual, bins=10):
    ref = pd.to_numeric(pd.Series(reference), errors='coerce').dropna().to_numpy(dtype=float)
    act = pd.to_numeric(pd.Series(actual), errors='coerce').dropna().to_numpy(dtype=float)
    if len(ref) < 5 or len(act) < 5:
        return float('nan')
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        if np.allclose(ref, ref[0]) and np.allclose(act, ref[0]):
            return 0.0
        center = float(np.median(ref))
        eps = max(1e-6, abs(center) * 1e-8)
        edges = np.array([-np.inf, center - eps, center + eps, np.inf])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, bins=edges)[0]/len(ref)
    a = np.histogram(act, bins=edges)[0]/len(act)
    smoothing = 1e-6
    return float(np.sum((a-r)*np.log((a+smoothing)/(r+smoothing))))


def categorical_chi2_pvalue(reference, actual):
    r = pd.Series(reference).fillna('Sin dato').astype(str).value_counts()
    a = pd.Series(actual).fillna('Sin dato').astype(str).value_counts()
    tab = pd.DataFrame({'referencia': r, 'actual': a}).fillna(0)
    if len(tab) < 2 or min(tab['referencia'].sum(), tab['actual'].sum()) < 5:
        return np.nan
    try:
        return float(chi2_contingency(tab.T)[1])
    except ValueError:
        return np.nan


def _nivel(psi, p):
    if np.isfinite(psi) and psi >= .25:
        return 'Rojo'
    if np.isfinite(psi) and psi >= .10:
        return 'Amarillo'
    if np.isfinite(p) and p < .01:
        return 'Rojo'
    return 'Verde'


def calculate_drift(reference_df, actual_df):
    """Compara datos históricos y nuevos; no necesita Pago_atiempo en los nuevos."""
    if reference_df.empty or actual_df.empty:
        raise ValueError('Referencia y actual deben contener registros')
    base = create_features(reference_df)
    now = create_features(actual_df)
    rows = []
    for col in MONITORED_NUMERIC:
        a = pd.to_numeric(base[col], errors='coerce').dropna()
        b = pd.to_numeric(now[col], errors='coerce').dropna()
        p = float(ks_2samp(a, b).pvalue) if len(a)>3 and len(b)>3 else np.nan
        psi = population_stability_index(a, b)
        level = _nivel(psi, p)
        # Una columna que desaparece en producción también constituye una alerta de calidad.
        null_ref = 100 * (1 - len(a) / len(base))
        null_current = 100 * (1 - len(b) / len(now))
        if len(a) >= 5 and (len(b) < 5 or null_current - null_ref > 20):
            level = 'Rojo'
        elif len(a) < 5 and len(b) < 5:
            level = 'Sin datos'
        rows.append({'variable': col, 'tipo': 'numerica', 'psi': psi, 'ks_pvalor': p,
                     'chi2_pvalor': np.nan, 'severidad': level,
                     'alerta': level in ('Rojo', 'Amarillo'),
                     'n_ref': int(len(a)), 'n_actual': int(len(b)),
                     'nulos_pct_ref': round(100*(1-len(a)/len(base)), 2),
                     'nulos_pct_actual': round(100*(1-len(b)/len(now)), 2)})
    for col in CATEGORICAL_COLS:
        p = categorical_chi2_pvalue(base[col], now[col])
        level = _nivel(float('nan'), p)
        rows.append({'variable': col, 'tipo': 'categorica', 'psi': np.nan,
                     'ks_pvalor': np.nan, 'chi2_pvalor': p,
                     'severidad': level, 'alerta': level in ('Rojo', 'Amarillo'),
                     'n_ref': len(base), 'n_actual': len(now),
                     'nulos_pct_ref': 0.0, 'nulos_pct_actual': 0.0})
    return pd.DataFrame(rows)


def demo_datasets(actual_path=None):
    historical = load_data()
    if actual_path is not None:
        return historical, pd.read_csv(actual_path, encoding='utf-8-sig'), False
    split = round(len(historical)*.7)
    return historical.iloc[:split], historical.iloc[split:], True


def save_report(result, real_data, source='CLI'):
    folder = ROOT/'artifacts'
    folder.mkdir(parents=True, exist_ok=True)
    result.to_csv(folder/'monitoreo_drift.csv', index=False)
    if real_data:
        row = pd.DataFrame([{'fecha_utc': datetime.now(timezone.utc).isoformat(),
                             'origen': source, 'variables': len(result),
                             'alertas': int(result['alerta'].sum()),
                             'alertas_rojas': int((result.severidad == 'Rojo').sum())}])
        history = folder/'historial_monitoreo.csv'
        row.to_csv(history, mode='a', header=not history.exists(), index=False)
    return folder/'monitoreo_drift.csv'


def main_cli():
    parser = argparse.ArgumentParser(description='Monitor de cambios en datos de crédito')
    parser.add_argument('--actual', help='CSV nuevo de clientes (NO requiere etiqueta)')
    args = parser.parse_args()
    ref, current, demo = demo_datasets(args.actual)
    result = calculate_drift(ref, current)
    path = save_report(result, real_data=not demo, source=str(args.actual) if args.actual else 'DEMO')
    if demo:
        print('DEMO: segmentos de una misma base; NO representa producción.')
    print(result.to_string(index=False)); print('CSV:', path)


def main_streamlit():
    import streamlit as st
    import matplotlib.pyplot as plt
    st.set_page_config(page_title='Monitoreo de créditos', layout='wide')
    st.title('Monitoreo de Data Drift — Créditos')
    st.caption('PSI 0.10–0.25 = advertencia; >=0.25 o p < 0.01 = alerta. Indicadores, no prueba de degradación.')
    uploaded = st.file_uploader('CSV con nuevos registros (sin Pago_atiempo)', type='csv')
    hist = load_data()
    if uploaded is None:
        n = round(len(hist)*.7)
        reference, actual = hist.iloc[:n], hist.iloc[n:]
        st.warning('MODO DEMO: se comparan dos segmentos del histórico. No son datos productivos.')
        is_real = False
    else:
        reference, actual = hist, pd.read_csv(uploaded, encoding='utf-8-sig')
        is_real = True
    try:
        result = calculate_drift(reference, actual)
    except ValueError as exc:
        st.error(str(exc)); return
    a,b,c = st.columns(3)
    a.metric('Variables observadas', len(result))
    b.metric('Advertencias y alertas', int(result.alerta.sum()))
    c.metric('Alertas rojas', int((result.severidad=='Rojo').sum()))
    if (result.severidad=='Rojo').any():
        st.error('🔴 Se detectaron cambios importantes; investigar calidad y población antes de reentrenar.')
    elif (result.severidad=='Amarillo').any():
        st.warning('🟡 Se detectaron cambios moderados.')
    else:
        st.success('🟢 Sin alertas bajo los umbrales establecidos.')
    st.subheader('Semáforo por variable')
    st.dataframe(result.sort_values(['alerta','psi'],ascending=[False,False]), use_container_width=True)
    psi = result[result.tipo=='numerica'].set_index('variable')['psi'].dropna().sort_values(ascending=False)
    st.bar_chart(psi)
    field = st.selectbox('Comparar distribuciones', MONITORED_NUMERIC)
    ra = pd.to_numeric(create_features(reference)[field], errors='coerce').dropna()
    rb = pd.to_numeric(create_features(actual)[field], errors='coerce').dropna()
    if len(ra) > 4 and len(rb) > 4:
        low, high = np.quantile(ra,[.01,.99])
        fig,ax=plt.subplots(figsize=(10,3.5))
        ax.hist(ra.clip(low,high),bins=25,alpha=.65,density=True,label='Histórico')
        ax.hist(rb.clip(low,high),bins=25,alpha=.5,density=True,label='Actual')
        ax.legend();ax.set_title(f'{field} — P1 a P99 de referencia')
        st.pyplot(fig);plt.close(fig)
    if is_real and st.button('Guardar informe e histórico de monitoreo'):
        st.success(f'Guardado: {save_report(result,True, "Streamlit")}')
    history = ROOT/'artifacts'/'historial_monitoreo.csv'
    if history.is_file():
        st.subheader('Evolución de alertas en ejecuciones reales')
        st.line_chart(pd.read_csv(history).set_index('fecha_utc')[['alertas','alertas_rojas']])
    st.info('Drift no significa automáticamente peor desempeño. Revisar métricas con etiquetas posteriores.')


if __name__ == '__main__':
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        via_streamlit = get_script_run_ctx(suppress_warning=True) is not None
    except ImportError:
        via_streamlit = False
    if via_streamlit:
        main_streamlit()
    else:
        main_cli()
