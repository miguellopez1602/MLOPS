pipeline {
    agent any
    options { timestamps() }
    triggers { pollSCM('H/15 * * * *') }
    stages {
        stage('Checkout') {
            steps { checkout scm }
        }
        stage('Comprobar estructura') {
            steps {
                sh '''
                  test -f Base_de_datos.csv
                  test -f requirements.txt
                  test -f mlops_pipeline/Cargar_datos.ipynb
                  test -f mlops_pipeline/comprension_eda.ipynb
                  test -f mlops_pipeline/ft_engineering.py
                  test -f mlops_pipeline/model_training_evaluation.py
                  test -f mlops_pipeline/model_deploy.py
                  test -f mlops_pipeline/model_monitoring.py
                '''
            }
        }
        stage('Instalar y probar') {
            steps {
                sh '''
                  python3 -m venv .venv
                  .venv/bin/python -m pip install --upgrade pip
                  .venv/bin/python -m pip install -r requirements.txt
                  .venv/bin/python -m pytest -q tests/
                '''
            }
        }
    }
    post {
        success { echo 'Pipeline completado: estructura y tests correctos.' }
        failure { echo 'Pipeline falló: revisar Console Output.' }
    }
}
