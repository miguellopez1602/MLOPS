Proyecto Final - Machine Learning

Estudiante: Miguel Ángel López Ardila  
Asignatura: Machine Learning  
Universidad: Universidad Católica Luis Amigó

Comprensión del negocio

El proyecto busca desarrollar un modelo de Machine Learning que permita predecir si un cliente pagará su crédito a tiempo, utilizando información financiera y características de los clientes.

El objetivo es identificar posibles riesgos de incumplimiento de pago, ayudando a las entidades financieras a tomar mejores decisiones al momento de otorgar créditos.

Para ello, se utiliza una base de datos de créditos, realizando procesos de limpieza, análisis exploratorio, preparación de datos, entrenamiento y evaluación de modelos predictivos.

Segunda entrega - Entrenamiento y evaluación

Se entrenaron y compararon modelos de clasificación, entre ellos regresión logística y Random Forest.

Durante la evaluación se encontró que aproximadamente el 95 % de los clientes pagaba sus créditos a tiempo. Debido a este desbalance, se utilizaron métricas como precisión, recall, F1, F2 y PR-AUC para evaluar el desempeño de los modelos.

Optimización del modelo

Se seleccionó Random Forest y se ajustó su umbral de decisión utilizando datos de validación, pasando de 0.50 a 0.20.

Los resultados obtenidos en el conjunto de prueba fueron:

| Métrica | Antes | Después |
|---|---|---|
| Precisión | 66,67 % | 10,66 % |
| Recall | 1,96 % | 33,33 % |
| F1-score | 3,81 % | 16,15 % |
| F2-score | 2,43 % | 23,38 % |

Justificación de los resultados

Inicialmente, el modelo identificaba solamente 2 de los 102 clientes que incumplieron. Después de ajustar el umbral, logró identificar 34.

Esto representa una mejora importante en la detección de incumplimientos, aunque también aumentaron las falsas alarmas a 285 casos.

Se decidió priorizar la identificación de clientes en riesgo, teniendo en cuenta que no detectar un posible incumplimiento puede generar pérdidas para una entidad financiera.

El ajuste permitió mejorar la detección sin cambiar el algoritmo seleccionado. Sin embargo, el modelo todavía necesita mejoras antes de utilizarse en decisiones financieras reales.
Proyecto Final - Machine Learning

 Miguel Ángel López Ardila  
Machine Learning  
Universidad Católica Luis Amigó

Comprensión del negocio

El proyecto busca desarrollar un modelo de Machine Learning que permita predecir si un cliente pagará su crédito a tiempo, utilizando información financiera y características de los clientes.

El objetivo es identificar posibles riesgos de incumplimiento de pago, ayudando a las entidades financieras a tomar mejores decisiones al momento de otorgar créditos.

Para ello, se utilizará una base de datos de créditos, realizando procesos de limpieza, análisis exploratorio, preparación de datos, entrenamiento y evaluación de modelos predictivos.
