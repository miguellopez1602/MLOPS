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


def _legacy_main_streamlit():
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



# ── Tablero visual ampliado ─────────────────────────────────────────────────
def _parse_credit_dates(series):
    """Acepta serial Excel con fracción horaria y textos ISO sin alterar el CSV."""
    serial = pd.to_numeric(series, errors='coerce')
    as_excel = pd.to_datetime(serial, unit='D', origin='1899-12-30', errors='coerce')
    as_text = pd.to_datetime(series.where(serial.isna()), errors='coerce')
    return as_excel.fillna(as_text)


def _js_from_probabilities(a, b):
    """Divergencia Jensen-Shannon, normalizada entre cero y uno (log base 2)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.sum() == 0 or b.sum() == 0:
        return np.nan
    a, b = a / a.sum(), b / b.sum()
    midpoint = (a + b) / 2
    def entropy_part(p):
        present = p > 0
        return np.sum(p[present] * np.log2(p[present] / midpoint[present]))
    return float(np.clip((entropy_part(a) + entropy_part(b)) / 2, 0, 1))


def _js_for_feature(col, kind, reference_features, actual_features):
    if kind == 'categorica':
        aa = reference_features[col].fillna('Sin dato').astype(str).value_counts()
        bb = actual_features[col].fillna('Sin dato').astype(str).value_counts()
        keys = aa.index.union(bb.index)
        return _js_from_probabilities(aa.reindex(keys, fill_value=0), bb.reindex(keys, fill_value=0))
    aa = pd.to_numeric(reference_features[col], errors='coerce').dropna().to_numpy()
    bb = pd.to_numeric(actual_features[col], errors='coerce').dropna().to_numpy()
    if len(aa) < 5 or len(bb) < 5:
        return np.nan
    edges = np.unique(np.quantile(aa, np.linspace(0, 1, 11)))
    if len(edges) < 3:
        middle = float(np.median(aa))
        deviation = max(1e-6, abs(middle) * 1e-8)
        edges = np.array([-np.inf, middle - deviation, middle + deviation, np.inf])
    else:
        edges[0], edges[-1] = -np.inf, np.inf
    return _js_from_probabilities(np.histogram(aa, bins=edges)[0], np.histogram(bb, bins=edges)[0])


def _report_for_dashboard(reference, current):
    """Conserva el reporte estadístico original y añade métricas visuales."""
    df = calculate_drift(reference, current).copy()
    ref_features, current_features = create_features(reference), create_features(current)
    js_values, ks_stats, psi_states = [], [], []
    for record in df.itertuples(index=False):
        js_values.append(_js_for_feature(record.variable, record.tipo, ref_features, current_features))
        if record.tipo == 'numerica':
            a = pd.to_numeric(ref_features[record.variable], errors='coerce').dropna()
            b = pd.to_numeric(current_features[record.variable], errors='coerce').dropna()
            ks_stats.append(float(ks_2samp(a, b).statistic) if len(a) > 3 and len(b) > 3 else np.nan)
            # PSI prioriza magnitud del cambio; KS p se muestra como prueba complementaria.
            if record.n_actual < 5 or (record.nulos_pct_actual - record.nulos_pct_ref) > 20:
                level = 'Rojo'
            elif np.isfinite(record.psi) and record.psi >= .25:
                level = 'Rojo'
            elif np.isfinite(record.psi) and record.psi >= .10:
                level = 'Amarillo'
            else:
                level = 'Verde' if np.isfinite(record.psi) else 'Sin datos'
        else:
            ks_stats.append(np.nan)
            jsv = js_values[-1]
            if not np.isfinite(jsv):
                level = 'Sin datos'
            elif jsv >= .10:
                level = 'Rojo'
            elif jsv >= .03:
                level = 'Amarillo'
            else:
                level = 'Verde'
        psi_states.append(level)
    df['ks_stat'] = ks_stats
    df['js'] = js_values
    df['estado_magnitud'] = psi_states
    return df


def _plot_psi_bars(data, top=15):
    import matplotlib.pyplot as plt
    d = data[(data['tipo'] == 'numerica') & data['psi'].notna()].nlargest(top, 'psi').iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, max(4, len(d) * .38 + .7)))
    colors = [{'Rojo':'#f43f72', 'Amarillo':'#f6c453', 'Verde':'#15c7af', 'Sin datos':'#8a99a8'}[s]
              for s in d['estado_magnitud']]
    ax.barh(d['variable'], d['psi'], color=colors)
    ax.axvline(.10, color='#f6c453', linestyle='--', linewidth=1.5, label='Advertencia: 0,10')
    ax.axvline(.25, color='#f43f72', linestyle='--', linewidth=1.5, label='Crítico: 0,25')
    ax.set_xlabel('Índice de estabilidad poblacional (PSI)')
    ax.set_title('Magnitud de cambio por variable', fontweight='bold', loc='left')
    ax.legend(loc='lower right')
    ax.grid(axis='x', alpha=.18)
    fig.tight_layout()
    return fig


def _plot_variable_comparison(reference, current, variable, is_categorical):
    import matplotlib.pyplot as plt
    a = create_features(reference)[variable]
    b = create_features(current)[variable]
    fig, ax = plt.subplots(figsize=(11, 4.1))
    if is_categorical:
        p = a.fillna('Sin dato').astype(str).value_counts(normalize=True)
        q = b.fillna('Sin dato').astype(str).value_counts(normalize=True)
        keys = p.add(q, fill_value=0).nlargest(10).index
        x = np.arange(len(keys))
        ax.bar(x - .2, p.reindex(keys, fill_value=0), .4, color='#37b9e4', label='Histórico')
        ax.bar(x + .2, q.reindex(keys, fill_value=0), .4, color='#f6a55f', label='Actual')
        ax.set_xticks(x, labels=keys, rotation=25, ha='right')
        ax.set_ylabel('Proporción de registros')
    else:
        aa = pd.to_numeric(a, errors='coerce').dropna()
        bb = pd.to_numeric(b, errors='coerce').dropna()
        if len(aa) >= 5 and len(bb) >= 5:
            low, high = np.quantile(aa, [.01, .99])
            if low == high:
                low -= 1
                high += 1
            ax.hist(aa.clip(low, high), bins=28, density=True, alpha=.70,
                    color='#37b9e4', label='Histórico')
            ax.hist(bb.clip(low, high), bins=28, density=True, alpha=.55,
                    color='#f6a55f', label='Actual')
            ax.set_ylabel('Densidad')
            ax.set_xlabel('Valores acotados a P1–P99 del histórico (solo visualización)')
        else:
            ax.text(.5, .5, 'Datos numéricos insuficientes para comparar',
                    transform=ax.transAxes, ha='center', va='center')
    ax.set_title(f'Histórico vs. actual · {variable}', fontweight='bold', loc='left')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=.18)
    fig.tight_layout()
    return fig


def _data_by_period(data, period):
    if data.empty or 'fecha_prestamo' not in data:
        return pd.DataFrame(columns=['periodo', 'registros', 'riesgo_real_pct'])
    dates = _parse_credit_dates(data['fecha_prestamo'])
    pcode = 'Q' if period == 'Trimestral' else 'M'
    valid = data.loc[dates.notna()].copy()
    valid['periodo'] = dates.loc[dates.notna()].dt.to_period(pcode).astype(str)
    valid['riesgo_real'] = np.nan
    if 'Pago_atiempo' in valid:
        labels = pd.to_numeric(valid['Pago_atiempo'], errors='coerce')
        valid['riesgo_real'] = (1 - labels).where(labels.isin([0, 1]))
    return (valid.groupby('periodo', as_index=False)
            .agg(registros=('periodo', 'size'), riesgo_real_pct=('riesgo_real', 'mean'))
            .assign(riesgo_real_pct=lambda d: d.riesgo_real_pct * 100))


def _format_percent(value):
    return f'{100*value:.1f} %' if isinstance(value, (int, float)) else '—'


def main_streamlit():
    import streamlit as st
    # Mantener compatibilidad con la prueba de la versión inicial, que simula
    # un Streamlit mínimo sin sidebar ni pestañas.
    if not hasattr(st, 'sidebar') or not hasattr(st, 'tabs'):
        return _legacy_main_streamlit()

    import json
    from datetime import date
    import matplotlib.pyplot as plt

    st.set_page_config(page_title='MLOPS | Observabilidad de crédito', page_icon='📊', layout='wide')
    st.markdown("""<style>
      .block-container {padding-top:1.5rem; max-width:1480px}
      [data-testid="stMetric"] {border:1px solid #45526955;border-radius:14px;padding:15px 18px;background:#27334a1a}
      [data-testid="stMetricValue"] {font-size:2rem}
      section[data-testid="stSidebar"] {border-right:1px solid #45526955}
      h1,h2,h3 {letter-spacing:-.025em}
    </style>""", unsafe_allow_html=True)
    st.title('📊 Centro de monitoreo — Riesgo crediticio')
    st.caption('MLOps · Detección de cambios en características, calidad de datos y seguimiento del modelo')

    historical = load_data()
    if 'fecha_prestamo' not in historical:
        st.error('El histórico necesita la columna fecha_prestamo para filtros temporales.')
        return
    dates = _parse_credit_dates(historical['fecha_prestamo'])
    min_date, max_date = dates.min().date(), dates.max().date()
    preferred = date(2025, 4, 1)
    initial = preferred if min_date <= preferred <= max_date else dates.quantile(.5).date()

    st.sidebar.header('⚙️ Parámetros')
    cut = st.sidebar.date_input('Fecha de corte (referencia / actual)',
                                value=initial, min_value=min_date, max_value=max_date)
    period = st.sidebar.radio('Periodicidad del muestreo', ['Mensual', 'Trimestral'], horizontal=False)
    minimum = st.sidebar.number_input('Mínimo de registros por período',
                                       min_value=5, max_value=5000, value=30, step=5)
    st.sidebar.divider()
    uploaded = st.sidebar.file_uploader('CSV actual de producción (opcional)', type=['csv'])
    is_demo = uploaded is None

    historical = historical.copy()
    historical['_fecha_monitor'] = dates
    cutoff = pd.Timestamp(cut)
    # Datos previos al corte como baseline; posteriores como conjunto actual.
    reference = historical.loc[historical['_fecha_monitor'] < cutoff].drop(columns=['_fecha_monitor'])
    if is_demo:
        actual = historical.loc[historical['_fecha_monitor'] >= cutoff].drop(columns=['_fecha_monitor'])
    else:
        try:
            actual = pd.read_csv(uploaded, encoding='utf-8-sig', low_memory=False)
        except Exception as exc:
            st.error(f'No se pudo leer el CSV: {exc}')
            return

    if reference.empty or actual.empty:
        st.warning('El filtro escogido dejó sin registros a la referencia o a la muestra actual. Cambia la fecha.')
        return
    if is_demo:
        st.warning('🧪 MODO DEMOSTRACIÓN: «actual» son registros de la misma base histórica posteriores al corte. No son datos de producción.')
    else:
        st.success('📥 CSV externo cargado. Se compara con los registros históricos anteriores al corte.')
        if 'fecha_prestamo' not in actual:
            st.info('El CSV actual no tiene fecha_prestamo: se analiza el drift, pero no la tendencia temporal del archivo.')

    try:
        report = _report_for_dashboard(reference, actual)
    except (ValueError, KeyError, TypeError) as exc:
        st.error(f'No se pudo calcular el monitoreo: {exc}')
        return
    red = report.loc[report.estado_magnitud=='Rojo', 'variable'].tolist()
    yellow = report.loc[report.estado_magnitud=='Amarillo', 'variable'].tolist()
    green = int((report.estado_magnitud=='Verde').sum())

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric('Registros referencia', f'{len(reference):,}')
    c2.metric('Registros actuales', f'{len(actual):,}')
    c3.metric('🔴 Variables críticas', len(red))
    c4.metric('🟡 En advertencia', len(yellow))
    c5.metric('🟢 Estables', green)

    st.subheader('🚨 Alertas y recomendaciones')
    if red:
        st.error('Drift alto o calidad insuficiente: ' + ', '.join(red[:8]) + ('…' if len(red)>8 else ''))
    if yellow:
        st.warning('Cambios moderados: ' + ', '.join(yellow[:8]) + ('…' if len(yellow)>8 else ''))
    if not red and not yellow:
        st.success('No se detectaron cambios relevantes en las magnitudes evaluadas.')
    st.info('Antes de reentrenar, verificar cambios en población, captura, nulos y rendimiento con etiquetas reales. Un p-valor pequeño por sí solo no implica drift crítico.')

    tab1, tab2, tab3, tab4 = st.tabs([
        '📋 Métricas por variable', '📈 Histórico vs. actual',
        '🗓️ Evolución temporal', '🎯 Desempeño del modelo'])

    with tab1:
        st.caption('Semáforo numérico: PSI < 0,10 verde · 0,10–0,25 amarillo · ≥ 0,25 rojo. Categóricas: divergencia JS < 0,03 verde · 0,03–0,10 amarillo · ≥ 0,10 rojo. Faltantes relevantes: rojo. KS y χ² son diagnósticos adicionales.')
        left, right = st.columns([2,1])
        with left:
            kind = st.selectbox('Filtrar tipo', ['Todas', 'Numéricas', 'Categóricas'], key='kind_filter')
        with right:
            show_only = st.checkbox('Solo alertas', value=False)
        shown = report.copy()
        if kind == 'Numéricas': shown = shown.loc[shown.tipo=='numerica']
        if kind == 'Categóricas': shown = shown.loc[shown.tipo=='categorica']
        if show_only: shown = shown.loc[shown.estado_magnitud.isin(['Rojo','Amarillo'])]
        shown = shown.sort_values(['alerta','psi'], ascending=[False, False], na_position='last')
        columns = ['variable','tipo','ks_stat','ks_pvalor','psi','js','chi2_pvalor',
                   'estado_magnitud','n_ref','n_actual','nulos_pct_ref','nulos_pct_actual']
        colors = {'Rojo':'background-color:#9e2542;color:white',
                  'Amarillo':'background-color:#a97e1a;color:white',
                  'Verde':'background-color:#087b62;color:white',
                  'Sin datos':'background-color:#5c6575;color:white'}
        style = shown[columns].style.format({k:'{:.3f}' for k in
            ['ks_stat','ks_pvalor','psi','js','chi2_pvalor','nulos_pct_ref','nulos_pct_actual']}, na_rep='—')
        style = style.map(lambda v: colors.get(v, ''), subset=['estado_magnitud'])
        st.dataframe(style, use_container_width=True, hide_index=True)
        st.download_button('⬇️ Descargar resultados (CSV)',
                           shown[columns].to_csv(index=False).encode('utf-8-sig'),
                           file_name='reporte_drift_filtrado.csv', mime='text/csv')
        fig = _plot_psi_bars(report)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with tab2:
        l, r = st.columns([2,1])
        with l:
            var = st.selectbox('Variable a comparar', MONITORED_NUMERIC + CATEGORICAL_COLS)
        with r:
            rec = report.loc[report.variable == var].iloc[0]
            st.metric('Estado', rec['estado_magnitud'])
            st.metric('PSI / JS', f"{rec['psi']:.3f}" if np.isfinite(rec['psi']) else f"{rec['js']:.3f}" if np.isfinite(rec['js']) else '—')
        fig = _plot_variable_comparison(reference, actual, var, var in CATEGORICAL_COLS)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        original = create_features(reference)[var]
        recent = create_features(actual)[var]
        if var in CATEGORICAL_COLS:
            a = original.fillna('Sin dato').astype(str).value_counts(normalize=True).head(10)
            b = recent.fillna('Sin dato').astype(str).value_counts(normalize=True).head(10)
            st.dataframe(pd.DataFrame({'Histórico %':a*100, 'Actual %':b*100}).fillna(0).round(2), use_container_width=True)
        else:
            summary = pd.DataFrame({'Histórico': pd.to_numeric(original,errors='coerce').describe(),
                                    'Actual': pd.to_numeric(recent,errors='coerce').describe()})
            st.dataframe(summary.round(2), use_container_width=True)

    with tab3:
        st.caption('Períodos que cumplen el mínimo de registros seleccionado; los datos anteriores al corte no se presentan como producción.')
        ref_period = _data_by_period(reference, period)
        actual_period = _data_by_period(actual, period)
        ref_period['grupo'] = 'Referencia'
        actual_period['grupo'] = 'Actual'
        merged = pd.concat([ref_period, actual_period], ignore_index=True)
        filtered = merged.loc[merged.registros >= minimum].copy()
        if filtered.empty:
            st.warning('No hay períodos con suficientes registros. Baja el mínimo en la barra lateral.')
        else:
            by_group = filtered.pivot_table(index='periodo',columns='grupo',values='registros',aggfunc='sum',fill_value=0)
            st.subheader('Volumen de solicitudes por período')
            st.bar_chart(by_group)
            risk_per = filtered.pivot_table(index='periodo', columns='grupo', values='riesgo_real_pct', aggfunc='mean')
            if risk_per.notna().any().any():
                st.subheader('Incumplimiento observado por período (cuando hay etiqueta)')
                st.line_chart(risk_per)
                st.caption('Este gráfico utiliza etiquetas históricas Pago_atiempo, NO predicciones del modelo.')
            st.dataframe(filtered[['periodo','grupo','registros','riesgo_real_pct']].round(2), use_container_width=True, hide_index=True)
        history_file = ROOT / 'artifacts' / 'historial_monitoreo.csv'
        if history_file.exists():
            history = pd.read_csv(history_file)
            st.subheader('Historial de ejecuciones del monitoreo real')
            if len(history) > 0 and 'fecha_utc' in history:
                st.line_chart(history.set_index('fecha_utc')[[x for x in ['alertas','alertas_rojas'] if x in history]])
        else:
            st.caption('Aún no hay evaluaciones reales guardadas. El historial se crea cuando se carga y guarda un CSV externo.')

    with tab4:
        st.caption('Resultados de la evaluación de prueba del modelo entrenado. NO son métricas de producción ni prueba de desempeño reciente.')
        manifest = ROOT / 'artifacts' / 'metricas_modelo.json'
        if not manifest.exists():
            st.warning('Falta metricas_modelo.json. Ejecuta el entrenamiento antes de presentar resultados.')
        else:
            try:
                meta = json.loads(manifest.read_text(encoding='utf-8'))
                m = meta.get('evaluacion_optimizada', {})
                x1,x2,x3,x4 = st.columns(4)
                x1.metric('Modelo seleccionado', meta.get('modelo','—'))
                x2.metric('Umbral óptimo', str(meta.get('umbral','—')))
                x3.metric('Recall (prueba)', _format_percent(m.get('recall_riesgo',0)))
                x4.metric('F2 (prueba)', _format_percent(m.get('f2_riesgo',0)))
                st.write('**Resultados en conjunto de prueba independiente**')
                key_map = {'precision_riesgo':'Precisión','recall_riesgo':'Recall',
                           'f1_riesgo':'F1','f2_riesgo':'F2','roc_auc':'ROC-AUC',
                           'pr_auc':'PR-AUC','accuracy':'Exactitud'}
                metric_table = pd.DataFrame({'Métrica':[key_map[k] for k in key_map if k in m],
                                              'Valor':[m[k] for k in key_map if k in m]})
                st.bar_chart(metric_table.set_index('Métrica'))
                img1,img2 = st.columns(2)
                for dest, filename, heading in [(img1,'matriz_confusion.png','Matriz de confusión'),
                                                 (img2,'curvas_roc_pr.png','Curvas ROC y precisión–recall')]:
                    pic = ROOT / 'artifacts' / filename
                    with dest:
                        if pic.exists():
                            st.image(str(pic), caption=heading, use_container_width=True)
                st.info('La precisión de incumplimiento es limitada. El modelo es académico, no apto para decisiones financieras reales.')
            except (ValueError, KeyError, OSError) as exc:
                st.error(f'No se pudieron cargar las métricas: {exc}')
        with st.expander('Comprobar salud de la API FastAPI'):
            endpoint = st.text_input('URL local de API', value='http://127.0.0.1:8002/health')
            if st.button('Consultar /health'):
                import urllib.request
                import urllib.error
                try:
                    with urllib.request.urlopen(endpoint, timeout=3) as response:
                        st.json(json.loads(response.read().decode('utf-8')))
                except Exception as exc:
                    st.warning(f'No se pudo conectar a la API: {exc}')

    if not is_demo:
        st.divider()
        if st.button('💾 Guardar evaluación de CSV externo en histórico', type='primary'):
            path = save_report(report.drop(columns=['estado_magnitud','js','ks_stat'],errors='ignore'),
                               real_data=True, source='Streamlit CSV externo')
            st.success(f'Reporte guardado en {path}. Historial actualizado.')
    st.caption('Interpretación: PSI y JS representan magnitud de cambio; KS y chi-cuadrado son pruebas complementarias. Drift no demuestra pérdida de calidad predictiva.')


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
