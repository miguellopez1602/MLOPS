# Estado de validación de las correcciones — 11/11/2026

## Ejecutado en el entorno de preparación

- Datos originales preservados (10.763 × 23).
- Los 2 notebooks se ejecutaron y se guardaron sin errores; `comprension_eda.ipynb` contiene 8 figuras PNG en sus salidas.
- Entrenamiento completo ejecutado con sklearn 1.8.0, pandas 2.2.3, numpy 2.3.5; modelo y JSON regenerados juntos.
- Modelo elegido: Random_forest_ajustado; umbral: 0.375; matriz de prueba: [[1912,139],[78,24]].
- API real `/health`: 200 listo; POST `/predict` con dos créditos: 200, cantidad=2, umbral=0.375.
- Pruebas: 20 aprobadas con `pytest`, cobertura global: 85 % bajo Python 3.13.5, dependencias numéricas fijadas.
- Data drift probado con DataFrame actual SIN etiqueta `Pago_atiempo`.

## Pendiente de validar en tu computador o servicios conectados

- `docker build -t ml-creditos-api .` con el Dockerfile nuevo de dos etapas (Docker no está disponible en este entorno).
- Jenkins: importar la nueva revisión del Jenkinsfile y obtener una ejecución SUCCESS con las 20 pruebas y un reporte JUnit.
- SonarQube: reejecutar el workflow con coverage.xml y comprobar el nuevo Quality Gate; el estado PASS de la versión anterior no garantiza el de la nueva.
- Notificaciones Jenkins por correo: requieren SMTP y `CI_NOTIFICATION_EMAIL` con correo real autorizado.
- Trigger automático por `pollSCM`: comprobar un build sin pulsar manualmente Construir ahora.

**La calificación corresponde al profesor. La presente revisión no representa una certificación de aptitud para crédito real.**
