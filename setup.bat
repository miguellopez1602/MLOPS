@echo off
setlocal
cd /d "%~dp0"
echo   PROYECTO MACHINE LEARNING - CREDITOS
where py >nul 2>&1
if %errorlevel%==0 (
  py -3 -m venv .venv
) else (
  python -m venv .venv
)
if errorlevel 1 (
  echo ERROR: No se pudo crear el entorno virtual. Instala Python 3.10 o superior.
  exit /b 1
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo ERROR: Fallo la instalacion de librerias.
  exit /b 1
)
echo LISTO. Usa .venv\Scripts\activate.bat para activar el entorno.
endlocal
