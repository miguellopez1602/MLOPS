# MLOPS — Predicción de riesgo crediticio

**Proyecto final de Machine Learning · Universidad Católica Luis Amigó · Miguel Ángel López Ardila**  
Repositorio principal: <https://github.com/miguellopez1602/MLOPS>

> **Uso educativo exclusivamente.** Las predicciones son experimentales: nunca se deben usar por sí solas para aprobar o rechazar créditos. No hay validación temporal, de calibración, de sesgos por grupo ni verificación de variables en producción.

## 1. Problema, dataset y criterio de éxito

Clasificación binaria de incumplimiento: `Pago_atiempo=0` significa **incumplimiento** y se transforma en clase positiva `riesgo=1`; `Pago_atiempo=1` es pago puntual y se transforma en `riesgo=0`.

El archivo `Base_de_datos.csv` contiene **10.763 registros y 23 columnas**, sin filas duplicadas completas. Hay **511 incumplimientos (4,75 %)** y 10.252 pagos puntuales. Se identificaron **7.175 valores nulos**. No se conoce con certeza la procedencia ni las condiciones de licencia de la fuente original; se conserva tal como fue facilitada para este ejercicio.

**Diccionario resumido de las 23 columnas:** `tipo_credito` (categoría), `fecha_prestamo` (serial Excel), `capital_prestado`, `plazo_meses`, `edad_cliente`, `tipo_laboral`, `salario_cliente`, `total_otros_prestamos`, `cuota_pactada`, `puntaje`, `puntaje_datacredito`, `cant_creditosvigentes`, `huella_consulta`, `saldo_mora`, `saldo_total`, `saldo_principal`, `saldo_mora_codeudor`, `creditos_sectorFinanciero`, `creditos_sectorCooperativo`, `creditos_sectorReal`, `promedio_ingresos_datacredito`, `tendencia_ingresos`, `Pago_atiempo`.

## 2. Instalación reproducible (Python 3.11)

Windows PowerShell desde la raíz de MLOPS:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

También se proporciona `setup.bat`. En Linux/macOS:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

`requirements-model.txt` fija exactamente la pila numérica usada en el reentrenamiento auditado (`scikit-learn 1.8.0`, `pandas 2.2.3`, `numpy 2.3.5`, `scipy 1.17.0`). `requirements-runtime.txt` añade API y `requirements-ci.txt` añade pruebas. La versión del sistema Python también debe ser compatible. Instalar versiones diferentes puede cambiar predicciones, umbrales y resultados.

## 3. EDA y creación de características

Abrir `mlops_pipeline/Cargar_datos.ipynb` y `mlops_pipeline/comprension_eda.ipynb` desde la raíz con JupyterLab. Ambos se ejecutaron y guardaron **sin errores**. El segundo incluye 8 figuras: proporción de clases, histogramas, boxplots, recuentos categóricos, `countplot`, boxplots por resultado, correlaciones Pearson y `pairplot` con `hue`. Reporta media, mediana, moda, desviación, rango, IQR, varianza, asimetría, curtosis y tipo de distribución.

`ft_engineering.py` genera **19 características** (13 numéricas, 3 categóricas y 3 derivadas); imputación, escalado y codificación se aprenden **dentro del pipeline ajustado en entrenamiento**. Por posible *data leakage*, se excluyen `puntaje`, `saldo_mora`, `saldo_total`, `saldo_principal` y `saldo_mora_codeudor`.

## 4. Entrenar y auditar el modelo

```powershell
.\.venv\Scripts\python.exe -m mlops_pipeline.model_training_evaluation
```

Se usa `random_state=42`, partición estratificada 80/20 y separación adicional del train en desarrollo/validación 75/25: **6.457 desarrollo, 2.153 validación, 2.153 test**. El conjunto de test se evalúa **solo después** de seleccionar el candidato por F2, sujeto a precisión mínima de 10 %, usando validación. Se comparan dos configuraciones de regresión logística y dos de Random Forest. También se realiza evaluación de consistencia con **3 pliegues en train**.

### Resultados de la ejecución reproducida

Modelo seleccionado: **`Random_forest_ajustado`**, umbral **`0.375`**. Esto sustituye el resultado histórico de umbral `0.20`, que correspondía a otra configuración/ejecución y no era reproducible de forma fiable con los scripts publicados.

| Métrica final en prueba | Umbral 0.50 | Umbral 0.375 |
|---|---:|---:|
| Accuracy | 95,12 % | 89,92 % |
| Precisión riesgo | 42,11 % | 14,72 % |
| Recall riesgo | 7,84 % | 23,53 % |
| F1 | 13,22 % | 18,11 % |
| F2 | 9,37 % | 21,02 % |
| ROC-AUC | 0,6447 | 0,6447 |
| PR-AUC | 0,1531 | 0,1531 |

En umbral 0.375 la matriz de confusión fue `[[1912, 139], [78, 24]]` (TN, FP, FN, TP). Se identificaron 24 de 102 casos de incumplimiento, con **139 falsos positivos**; no se debe confundir *accuracy* elevado con una buena detección de la clase minoritaria. Con 3 pliegues estratificados de entrenamiento, PR-AUC fue **0,1356 ± 0,0013**; esto solo mide consistencia interna, no generalización temporal. La elección responde a F2 y una restricción de precisión; el coste económico de FN/FP no está cuantificado.

**Trazabilidad:** `artifacts/metricas_modelo.json` conserva semilla, versiones, hashes SHA256, precisión, recall, F1, F2, accuracy, AUC, confusión, CV y latencias. `modelo_creditos.joblib` se genera con el mismo entrenamiento. `busqueda_umbrales.csv`, `comparacion_modelos.csv`, `comparacion_modelos.png`, `comparacion_antes_despues.png`, `matriz_confusion.png`, `curvas_roc_pr.png` documentan cada decisión. El gráfico de comparación usa **solo umbrales que cumplen el piso de precisión**.

## 5. API FastAPI — JSON y lotes

```powershell
.\.venv\Scripts\python.exe -m uvicorn mlops_pipeline.model_deploy:app --host 127.0.0.1 --port 8000
```

Documentación: <http://127.0.0.1:8000/docs>. GET `/health` comprueba carga y consistencia modelo/JSON; POST `/predict` valida estrictamente 1–1000 registros y devuelve probabilidades y clases en JSON. **No admite entradas vacías, edades imposibles ni `puntaje`/saldos de mora.**

Ejemplo para POST `/predict`:

```json
{
  "registros": [
    {"capital_prestado": 2000000, "plazo_meses": 12, "edad_cliente": 30,
     "salario_cliente": 4000000, "cuota_pactada": 400000,
     "tipo_credito": 4, "tipo_laboral": "Empleado", "tendencia_ingresos": "Estable"},
    {"capital_prestado": 10000000, "plazo_meses": 24, "edad_cliente": 45,
     "salario_cliente": 1500000, "cuota_pactada": 800000,
     "tipo_credito": 9, "tipo_laboral": "Independiente", "tendencia_ingresos": "Decreciente"}
  ]
}
```

También hay un JSON de prueba en `examples/prediccion_batch.json`. Los campos mínimos están tipados en `CreditRecord`. Si falta un campo requerido responde **HTTP 422**. Si faltan o no corresponden los artefactos responde **503** en inferencia; `/health` marca `no_listo`.

## 6. Docker desde una clonación limpia

**No necesita copiar un modelo previamente entrenado**: el `Dockerfile` tiene dos etapas. La primera entrena un modelo reproducible a partir del CSV y dependencias fijadas; la segunda publica solo la API y sus artefactos, usando un usuario sin privilegios. `artifacts/*.joblib` se ignora en Git por tamaño; el binario se crea automáticamente dentro del Docker build.

```powershell
docker build -t ml-creditos-api .
docker run --rm -p 8001:8000 --name ml-creditos-api ml-creditos-api
```

Prueba: <http://localhost:8001/health> y <http://localhost:8001/docs>. El build puede consumir CPU durante el entrenamiento y requiere acceso a los índices de paquetes de Python. Evitar exponer este ejemplo fuera de `localhost` sin autenticarlo.

## 7. Data Drift (Streamlit y CLI)

```powershell
.\.venv\Scripts\python.exe -m streamlit run mlops_pipeline/model_monitoring.py
.\.venv\Scripts\python.exe mlops_pipeline/model_monitoring.py
.\.venv\Scripts\python.exe mlops_pipeline/model_monitoring.py --actual nuevos_creditos.csv
```

Hay un CSV sintético para experimentar: `examples/nuevos_creditos_demo.csv`. Sin `--actual` o archivo subido, se indica explícitamente **modo demo**. Para datos nuevos no se exige `Pago_atiempo` porque aún podría desconocerse. Se comparan distribuciones con PSI, KS, chi-cuadrado, semáforo por variable y gráficos antes/después. Cuando se envían datos reales se guarda un histórico de alertas en `artifacts/historial_monitoreo.csv` (desde CLI automáticamente; en la interfaz al pulsar Guardar). **Drift no demuestra automáticamente deterioro del rendimiento del modelo.**

## 8. Pruebas, cobertura, GitHub Actions y SonarQube

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests --cov=mlops_pipeline --cov-report=xml:coverage.xml
```

La suite de **20 pruebas** valida ratios, exclusión de fuga de información, estratificación, métricas, drift, generación integrada de artefactos, integridad SHA256, interfaz Streamlit y contratos positivos/negativos de API. En el entorno de auditoría se midió **85 % de cobertura de código Python** con `pytest-cov` (el resultado puede variar al instalar otros paquetes). GitHub Actions instala Python 3.11, ejecuta tests y luego el escáner de SonarQube Cloud. Configura `SONAR_TOKEN` como **GitHub repository secret**, nunca como texto en el repositorio. SonarQube lee `coverage.xml` mediante `sonar.python.coverage.reportPaths`.

## 9. Jenkins — integración continua

Para montar Jenkins local (se requiere Docker Desktop):

```powershell
docker build -f Dockerfile.jenkins -t jenkins-mlops-python .
docker run -d --name jenkins-mlops -p 127.0.0.1:8085:8080 -v jenkins_mlops_home:/var/jenkins_home jenkins-mlops-python
```

Si ya tienes `jenkins-mlops` configurado, **no crees un segundo contenedor con el mismo nombre**: conserva `jenkins_mlops_home` y actualiza la imagen solo si es necesario. Abre <http://localhost:8085>. Crea tarea **Pipeline → Pipeline script from SCM → Git**, repositorio `https://github.com/miguellopez1602/MLOPS.git`, rama `*/master`, script `Jenkinsfile`. El pipeline comprueba estructura, instala dependencias de CI, ejecuta tests JUnit y cobertura y conserva informes. Configura sondeo `pollSCM` cada aproximadamente 15 minutos. Debe comprobarse un disparo automático posterior, no basta una ejecución manual.

**Notificaciones externas:** el `Jenkinsfile` incluye `mail` condicional: configura SMTP de Jenkins y una variable de entorno `CI_NOTIFICATION_EMAIL` (destinatario). Si no está configurada, el pipeline no envía correo y lo indica en consola; no se ha validado SMTP desde este entorno. Nunca publicar credenciales de Jenkins o SonarQube.

Para un chequeo integrado en Windows: `powershell -ExecutionPolicy Bypass -File .\VERIFICAR_EN_WINDOWS.ps1`. La guía completa de hallazgos y sus evidencias está en `docs/ESTADO_VALIDACION.md`; la ficha técnica de riesgo del modelo en `docs/model_card.md`.

## 10. Estrategia Git y limitaciones

Ramas solicitadas: **`developer` → `certification` → `master`**. Crear correcciones en una rama `feature/...`, probar y promover en ese orden mediante PR o fast-forward cuando proceda.

**Limitaciones no resueltas por una demo:** calibración real de probabilidades, generalización temporal, sesgos por grupo, cambio de contexto económico, etiquetas futuras, fraude y explicación del rechazo. La clase positiva es infrecuente y la precisión de ~15 % impide utilizar alertas de forma autónoma. Para producción se necesita gobierno de datos y evaluación adicional.
