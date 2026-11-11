
import pandas as pd

from mlops_pipeline.ft_engineering import (
    create_features,
    FEATURE_COLS,
)


def test_creacion_de_caracteristicas():
    datos = pd.DataFrame([{
        "capital_prestado": 2000000,
        "salario_cliente": 4000000,
        "cuota_pactada": 400000,
        "edad_cliente": 30,
        "plazo_meses": 12,
        "tipo_credito": 4,
        "tipo_laboral": "Empleado",
        "tendencia_ingresos": "Estable",
    }])

    resultado = create_features(datos)

    assert list(resultado.columns) == FEATURE_COLS
    assert resultado["ratio_cuota_salario"].iloc[0] == 0.1
    assert resultado["ratio_capital_salario"].iloc[0] == 0.5


def test_exclusion_de_fuga_de_datos():
    datos = pd.DataFrame([{
        "capital_prestado": 1000000,
        "salario_cliente": 2000000,
        "saldo_mora": 500000,
        "puntaje": 90,
    }])

    resultado = create_features(datos)

    assert "saldo_mora" not in resultado.columns
    assert "puntaje" not in resultado.columns
