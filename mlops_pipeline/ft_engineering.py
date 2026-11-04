from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "Base_de_datos.csv"
TARGET = "Pago_atiempo"

# Se excluyen saldos actuales y de mora, por riesgo de fuga temporal:
# Además, puntaje separa PERFECTAMENTE las clases (max riesgo 62.67 vs min no riesgo 63.81),
LEAKAGE_CANDIDATES = ["saldo_mora", "saldo_total", "saldo_principal", "saldo_mora_codeudor", "puntaje"]
NUMERIC_COLS = [
    "capital_prestado", "plazo_meses", "edad_cliente", "salario_cliente",
    "total_otros_prestamos", "cuota_pactada", "puntaje_datacredito",
    "cant_creditosvigentes", "huella_consulta", "creditos_sectorFinanciero",
    "creditos_sectorCooperativo", "creditos_sectorReal",
    "promedio_ingresos_datacredito",
]
CATEGORICAL_COLS = ["tipo_credito", "tipo_laboral", "tendencia_ingresos"]
DERIVED_NUMERIC = ["mes_prestamo", "ratio_cuota_salario", "ratio_capital_salario"]
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS + DERIVED_NUMERIC


def load_data(path=DATA_PATH):
    """Lee CSV y comprueba la existencia del objetivo."""
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    if TARGET not in df.columns:
        raise ValueError(f"No se encontró la variable objetivo {TARGET}")
    return df


def create_features(df):
    """Normaliza tipos, faltantes y crea variables disponibles al solicitar crédito.

    NO calcula estadísticos globales para imputar: eso se aprende SOLO en entrenamiento
    dentro de los Pipelines de scikit-learn.
    """
    work = df.copy()
    work = work.replace(r"^\s*$", np.nan, regex=True)
    for c in NUMERIC_COLS:
        if c not in work.columns:
            work[c] = np.nan
        work[c] = pd.to_numeric(work[c], errors="coerce")
    for c in CATEGORICAL_COLS:
        if c not in work.columns:
            work[c] = pd.NA

    work["edad_cliente"] = work["edad_cliente"].where(work["edad_cliente"].between(18, 100))
    # Valores superiores a 100 millones son atípicos y probablemente errores de captura.
    work["salario_cliente"] = work["salario_cliente"].where(
        work["salario_cliente"].between(1, 100_000_000)
    )
    work["puntaje_datacredito"] = work["puntaje_datacredito"].where(
        work["puntaje_datacredito"].between(1, 1000)
    )
    for c in ["capital_prestado", "cuota_pactada", "total_otros_prestamos",
              "promedio_ingresos_datacredito"]:
        work[c] = work[c].where(work[c].ge(0))
    work["plazo_meses"] = work["plazo_meses"].where(work["plazo_meses"].between(1, 360))

    work["tipo_credito"] = (
        pd.to_numeric(work["tipo_credito"], errors="coerce")
        .astype("Int64").astype("string").fillna("Sin dato").astype(str)
    )
    work["tipo_laboral"] = (work["tipo_laboral"].astype("string")
                            .str.strip().str.title().fillna("Sin dato").astype(str))
    trend = work["tendencia_ingresos"].astype("string").str.strip().str.title()
    work["tendencia_ingresos"] = trend.where(
        trend.isin(["Creciente", "Estable", "Decreciente"]), "Sin dato"
    ).fillna("Sin dato").astype(str)

    # fecha_prestamo viene como número de serie
    if "fecha_prestamo" in work.columns:
        raw = work["fecha_prestamo"]
        serial = pd.to_numeric(raw, errors="coerce")
        date_from_serial = pd.to_datetime(serial, unit="D", origin="1899-12-30", errors="coerce")
        date_from_text = pd.to_datetime(raw.where(serial.isna()), errors="coerce")
        date = date_from_serial.fillna(date_from_text)
        work["mes_prestamo"] = date.dt.month.astype(float)
    else:
        work["mes_prestamo"] = np.nan

    salary = work["salario_cliente"].replace({0: np.nan})
    work["ratio_cuota_salario"] = (work["cuota_pactada"] / salary).replace([np.inf, -np.inf], np.nan)
    work["ratio_capital_salario"] = (work["capital_prestado"] / salary).replace([np.inf, -np.inf], np.nan)
    # Un tope evita que valores imposibles dominen el modelo.
    for c in ["ratio_cuota_salario", "ratio_capital_salario"]:
        work[c] = work[c].clip(0, 100)
    return work[FEATURE_COLS]


def prepare_training_data(df=None, test_size=0.20, random_state=42):
    """Separa datos y etiquetas sin permitir que el objetivo esté en X."""
    if df is None:
        df = load_data()
    target = pd.to_numeric(df[TARGET], errors="coerce")
    valid = target.isin([0, 1])
    X = create_features(df.loc[valid])
    y = (1 - target.loc[valid].astype(int)).astype(int)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def build_preprocessor():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("numericas", numeric, NUMERIC_COLS + DERIVED_NUMERIC),
        ("categoricas", categorical, CATEGORICAL_COLS),
    ])


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te = prepare_training_data()
    print("Entrenamiento:", X_tr.shape, "Prueba:", X_te.shape)
    print("Incumplimiento en train (%):", round(y_tr.mean() * 100, 2))
    print("Excluidas por posible fuga:", LEAKAGE_CANDIDATES)
