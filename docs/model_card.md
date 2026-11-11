# Ficha del modelo — Incumplimiento de créditos

**Versión académica:** 2026-11-11 | **Algoritmo:** Random Forest ajustado | **Umbral elegido:** 0.375

## Propósito y población
Identificar posibles créditos con incumplimiento de pago para una demostración de aprendizaje supervisado. Datos originales: 10.763 créditos, 511 incumplimientos (4,75 %). El origen, período de recolección y permisos de la fuente no se encuentran documentados; **no se presupone autorización para tomar decisiones reales**.

## Variable de resultado
`Pago_atiempo` es 1 si el cliente pagó a tiempo, 0 si incumplió. Para la API y el modelo se invierte el objetivo: clase positiva `1 = riesgo de incumplimiento`.

## Entradas
13 características numéricas, 3 categóricas y 3 derivadas de cuota/capital/salario y fecha. Se excluyen `puntaje` y cuatro columnas de saldos/mora por posible fuga de información. Se rechazan registros con salario, edad, plazo, capital, cuota, crédito o empleo faltantes o inválidos en la API.

## Entrenamiento y criterio de elección
80 % entrenamiento y 20 % prueba con estratificación; el entrenamiento se separa internamente 75/25 para seleccionar el modelo/umbral. Comparación de 4 candidatos de 2 familias con probabilidad, F2 (pondera recall), PR-AUC y una exigencia mínima de precisión del 10 %. La matriz de prueba se consulta solo una vez para evaluar el ganador. Validación cruzada en train con 3 pliegues estima PR-AUC y su desviación; no se usa test para elegir modelo.

## Resultados de la reproducción
`Random_forest_ajustado`, umbral `0.375`: TN 1.912; FP 139; FN 78; TP 24; precisión 0.1472; recall 0.2353; F1 0.1811; F2 0.2102; ROC-AUC 0.6447; PR-AUC 0.1531; exactitud 0.8992. PR-AUC validación cruzada (3 pliegues de train): media 0.1356 y desviación 0.0013.

## Consistencia, rendimiento y escalabilidad
La desviación en tres pliegues es pequeña en este conjunto, pero no asegura estabilidad entre épocas, regiones ni políticas crediticias. El entrenamiento se hace en ~10.763 registros y la API procesa hasta 1.000 solicitudes por lote; la capacidad real requiere ensayos de concurrencia/carga para un entorno productivo. La duración del entrenamiento y latencia de inferencia se guardan en `artifacts/metricas_modelo.json`, dependen del hardware y deben evitarse afirmaciones de escalabilidad no medidas.

## Limitaciones y uso responsable
La precisión baja implica una alta proporción de falsos positivos en alertas. Las probabilidades no están calibradas ni se ha validado un período futuro independiente. Variables crediticias pueden correlacionarse con atributos sensibles; no se midió equidad por grupos. Se debe verificar la disponibilidad temporal de variables y condiciones de consentimiento del dataset. El modelo **no se debe usar para rechazar créditos ni automatizar decisiones sobre personas**.

## Artefactos y trazabilidad
Modelo joblib y manifiesto JSON tienen SHA256 enlazado para detectar mezclas de versiones; el Dockerfile los genera desde una clonación limpia. La versión de sklearn y el hash del dataset constan en el manifiesto. La suite de tests verifica funcionalidad, no exactitud financiera de datos nuevos.
