"""Preparación sin fuga temporal de información para riesgo de incumplimiento."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / 'Base_de_datos.csv'
TARGET = 'Pago_atiempo'  # 1 = pagó, 0 = incumplió. En el modelo riesgo = 1.
LEAKAGE_CANDIDATES = ['saldo_mora', 'saldo_total', 'saldo_principal', 'saldo_mora_codeudor', 'puntaje']
NUMERIC_COLS = [
    'capital_prestado', 'plazo_meses', 'edad_cliente', 'salario_cliente',
    'total_otros_prestamos', 'cuota_pactada', 'puntaje_datacredito',
    'cant_creditosvigentes', 'huella_consulta', 'creditos_sectorFinanciero',
    'creditos_sectorCooperativo', 'creditos_sectorReal',
    'promedio_ingresos_datacredito',
]
CATEGORICAL_COLS = ['tipo_credito', 'tipo_laboral', 'tendencia_ingresos']
DERIVED_NUMERIC = ['mes_prestamo', 'ratio_cuota_salario', 'ratio_capital_salario']
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS + DERIVED_NUMERIC
# Esquema mínimo de inferencia. Otros campos son opcionales y admiten imputación.
REQUIRED_INPUTS = ['capital_prestado', 'plazo_meses', 'edad_cliente',
                   'salario_cliente', 'cuota_pactada', 'tipo_credito', 'tipo_laboral']


def load_data(path=DATA_PATH):
    """Carga el histórico ETIQUETADO para entrenamiento."""
    df = pd.read_csv(path, encoding='utf-8-sig', low_memory=False)
    missing = set(REQUIRED_INPUTS + [TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f'Faltan columnas en histórico: {sorted(missing)}')
    valid = pd.to_numeric(df[TARGET], errors='coerce').dropna().isin([0, 1]).all()
    if not valid:
        raise ValueError('Pago_atiempo debe contener solamente 0 o 1')
    return df


def create_features(df):
    """Normaliza datos y deriva variables; no aprende medianas ni categorías globales.

    La imputación y codificación se ajustan exclusivamente con datos de entrenamiento.
    Esta función también acepta solicitudes sin etiqueta objetivo para API/monitoreo.
    """
    work = df.copy().replace(r'^\s*$', np.nan, regex=True)
    for c in NUMERIC_COLS:
        if c not in work:
            work[c] = np.nan
        work[c] = pd.to_numeric(work[c], errors='coerce')
    for c in CATEGORICAL_COLS:
        if c not in work:
            work[c] = pd.NA

    work['edad_cliente'] = work['edad_cliente'].where(work['edad_cliente'].between(18, 100))
    work['salario_cliente'] = work['salario_cliente'].where(work['salario_cliente'].between(1, 100_000_000))
    work['puntaje_datacredito'] = work['puntaje_datacredito'].where(work['puntaje_datacredito'].between(1, 1000))
    work['plazo_meses'] = work['plazo_meses'].where(work['plazo_meses'].between(1, 360))
    for c in ['capital_prestado', 'cuota_pactada', 'total_otros_prestamos',
              'promedio_ingresos_datacredito', 'cant_creditosvigentes', 'huella_consulta',
              'creditos_sectorFinanciero', 'creditos_sectorCooperativo', 'creditos_sectorReal']:
        work[c] = work[c].where(work[c].ge(0))

    work['tipo_credito'] = (pd.to_numeric(work['tipo_credito'], errors='coerce')
                            .astype('Int64').astype('string').fillna('Sin dato').astype(str))
    work['tipo_laboral'] = (work['tipo_laboral'].astype('string').str.strip().str.title()
                             .fillna('Sin dato').astype(str))
    trend = work['tendencia_ingresos'].astype('string').str.strip().str.title()
    work['tendencia_ingresos'] = (trend.where(trend.isin(['Creciente', 'Estable', 'Decreciente']), 'Sin dato')
                                  .fillna('Sin dato').astype(str))

    if 'fecha_prestamo' in work.columns:
        raw = work['fecha_prestamo']
        serial = pd.to_numeric(raw, errors='coerce')
        from_serial = pd.to_datetime(serial, unit='D', origin='1899-12-30', errors='coerce')
        from_text = pd.to_datetime(raw.where(serial.isna()), errors='coerce')
        work['mes_prestamo'] = from_serial.fillna(from_text).dt.month.astype(float)
    else:
        work['mes_prestamo'] = np.nan
    salary = work['salario_cliente'].replace(0, np.nan)
    work['ratio_cuota_salario'] = (work['cuota_pactada'] / salary).replace([np.inf, -np.inf], np.nan).clip(0, 100)
    work['ratio_capital_salario'] = (work['capital_prestado'] / salary).replace([np.inf, -np.inf], np.nan).clip(0, 100)
    return work[FEATURE_COLS]


def prepare_training_data(df=None, test_size=0.20, random_state=42):
    if df is None:
        df = load_data()
    target = pd.to_numeric(df[TARGET], errors='coerce')
    valid = target.isin([0, 1])
    X = create_features(df.loc[valid])
    y = (1 - target.loc[valid].astype(int)).astype(int)
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def build_preprocessor():
    numeric = Pipeline([
        ('imputer', SimpleImputer(strategy='median', add_indicator=True)),
        ('scaler', StandardScaler()),
    ])
    categorical = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
    ])
    return ColumnTransformer([
        ('numericas', numeric, NUMERIC_COLS + DERIVED_NUMERIC),
        ('categoricas', categorical, CATEGORICAL_COLS),
    ])


if __name__ == '__main__':
    X_tr, X_te, y_tr, y_te = prepare_training_data()
    print('Train:', X_tr.shape, 'Test:', X_te.shape)
    print('Riesgo en train:', round(y_tr.mean(), 4))
    print('Variables excluidas por fuga:', LEAKAGE_CANDIDATES)
