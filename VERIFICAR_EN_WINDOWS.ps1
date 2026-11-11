# Ejecutar desde la carpeta MLOPS corregida en PowerShell.
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$python = '.\.venv\Scripts\python.exe'
if (-not (Test-Path $python)) {
    py -3.11 -m venv .venv
}
& $python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'Falló instalación de requirements' }
& $python -m pytest -q tests --cov=mlops_pipeline --cov-report=xml:coverage.xml
if ($LASTEXITCODE -ne 0) { throw 'Fallaron las pruebas' }
& $python -m mlops_pipeline.model_training_evaluation
if ($LASTEXITCODE -ne 0) { throw 'Falló entrenamiento' }
& $python -m compileall -q mlops_pipeline
Write-Host 'Validación MLOPS completada. Consulta artifacts/metricas_modelo.json.'
