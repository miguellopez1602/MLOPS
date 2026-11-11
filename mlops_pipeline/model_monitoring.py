import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
try:
    from .ft_engineering import load_data, create_features, ROOT, NUMERIC_COLS, CATEGORICAL_COLS
except ImportError:
    from ft_engineering import load_data, create_features, ROOT, NUMERIC_COLS, CATEGORICAL_COLS


def population_stability_index(reference, actual, bins=10):
    """PSI con límites construidos solo a partir de la referencia."""
    ref = pd.to_numeric(pd.Series(reference), errors="coerce").dropna().to_numpy()
    act = pd.to_numeric(pd.Series(actual), errors="coerce").dropna().to_numpy()
    if len(ref) < 5 or len(act) < 5:
        return np.nan
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0 if np.nanmedian(ref) == np.nanmedian(act) else np.nan
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, bins=edges)[0] / len(ref)
    a = np.histogram(act, bins=edges)[0] / len(act)
    eps = 1e-6
    return float(np.sum((a-r) * np.log((a+eps)/(r+eps))))


def categorical_chi2_pvalue(reference, actual):
    r = pd.Series(reference).fillna("Sin dato").astype(str).value_counts()
    a = pd.Series(actual).fillna("Sin dato").astype(str).value_counts()
    tab = pd.DataFrame({"ref": r, "actual": a}).fillna(0)
    if len(tab) < 2 or tab["ref"].sum() == 0 or tab["actual"].sum() == 0:
        return np.nan
    return float(chi2_contingency(tab.T)[1])


def calculate_drift(reference_df, actual_df):
    baseline, current = create_features(reference_df), create_features(actual_df)
    report = []
    for col in NUMERIC_COLS + ["ratio_cuota_salario", "ratio_capital_salario"]:
        a = pd.to_numeric(baseline[col], errors="coerce").dropna()
        b = pd.to_numeric(current[col], errors="coerce").dropna()
        p = float(ks_2samp(a, b).pvalue) if len(a) > 3 and len(b) > 3 else np.nan
        psi = population_stability_index(a, b)
        report.append({"variable": col, "tipo": "numerica", "psi": psi,
                       "ks_pvalor": p, "chi2_pvalor": np.nan,
                       "alerta": bool((np.isfinite(psi) and psi >= 0.2) or (np.isfinite(p) and p < 0.01))})
    for col in CATEGORICAL_COLS:
        p = categorical_chi2_pvalue(baseline[col], current[col])
        report.append({"variable": col, "tipo": "categorica", "psi": np.nan,
                       "ks_pvalor": np.nan, "chi2_pvalor": p,
                       "alerta": bool(np.isfinite(p) and p < 0.01)})
    return pd.DataFrame(report)


def demo_datasets(actual_path=None):
    hist = load_data()
    if actual_path is not None:
        return hist, load_data(actual_path), False
    # Para aprendizaje, no se usa como evidencia de data drift real.
    pivot = round(len(hist) * .7)
    return hist.iloc[:pivot], hist.iloc[pivot:], True


def main_cli():
    p = argparse.ArgumentParser()
    p.add_argument("--actual", help="CSV de registros NUEVOS de producción")
    args = p.parse_args()
    ref, actual, demo = demo_datasets(args.actual)
    out = calculate_drift(ref, actual)
    path = ROOT / "artifacts" / "monitoreo_drift.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    if demo:
        print("DEMO: referencia y actual provienen de la misma base histórica (NO producción).")
    print(out.to_string(index=False))
    print("Resultado:", path)


def main_streamlit():
    import streamlit as st
    import matplotlib.pyplot as plt
    st.set_page_config(page_title="Monitoreo de créditos", layout="wide")
    st.title("Monitoreo de Data Drift - Créditos")
    st.caption("Los umbrales PSI >= 0.2 o p < 0.01 son alertas orientativas, no certeza de deterioro del modelo.")
    uploaded = st.file_uploader("CSV con datos actuales de producción (opcional)", type="csv")
    reference = load_data()
    if uploaded is None:
        split = round(len(reference) * .7)
        a, b = reference.iloc[:split], reference.iloc[split:]
        st.warning("Modo demostración: se comparan dos segmentos de la misma base histórica.")
    else:
        a, b = reference, pd.read_csv(uploaded)
    result = calculate_drift(a, b)
    c1, c2 = st.columns(2)
    c1.metric("Variables analizadas", len(result))
    c2.metric("Variables con alertas", int(result["alerta"].sum()))
    st.dataframe(result, use_container_width=True)
    col = st.selectbox("Comparar una variable", NUMERIC_COLS)
    fig, ax = plt.subplots()
    vals_a = pd.to_numeric(create_features(a)[col], errors="coerce").dropna()
    vals_b = pd.to_numeric(create_features(b)[col], errors="coerce").dropna()
    bounds = np.quantile(vals_a, [0.01, 0.99])
    ax.hist(vals_a.clip(*bounds), bins=25, alpha=.6, density=True, label="Referencia")
    ax.hist(vals_b.clip(*bounds), bins=25, alpha=.6, density=True, label="Actual")
    ax.legend(); ax.set_title(col)
    st.pyplot(fig)
    st.info("Si aumenta el drift, revisar la calidad de nuevos registros, la población y el rendimiento del modelo antes de reentrenar.")


if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        is_streamlit = get_script_run_ctx(suppress_warning=True) is not None
    except ImportError:
        is_streamlit = False
    if is_streamlit:
        main_streamlit()
    else:
        main_cli()
