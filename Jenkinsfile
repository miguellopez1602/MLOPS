pipeline {
    agent any
    options {
        timestamps()
        skipDefaultCheckout(true)
        disableConcurrentBuilds()
        buildDiscarder(logRotator(numToKeepStr: '15'))
    }
    triggers { pollSCM('H/15 * * * *') }
    stages {
        stage('Checkout GitHub') {
            steps { checkout scm }
        }
        stage('Verificar estructura') {
            steps {
                sh '''
                  set -eu
                  for f in Base_de_datos.csv requirements.txt requirements-ci.txt \
                      mlops_pipeline/Cargar_datos.ipynb mlops_pipeline/comprension_eda.ipynb \
                      mlops_pipeline/ft_engineering.py mlops_pipeline/model_training_evaluation.py \
                      mlops_pipeline/model_deploy.py mlops_pipeline/model_monitoring.py \
                      tests/test_pipeline.py Dockerfile; do test -f "$f"; done
                '''
            }
        }
        stage('Pruebas y cobertura') {
            steps {
                sh '''
                  set -eu
                  python3 -m venv .venv
                  .venv/bin/python -m pip install -q -r requirements-ci.txt
                  .venv/bin/python -m pytest -q tests --junitxml=junit.xml \
                    --cov=mlops_pipeline --cov-report=xml:coverage.xml --cov-report=term
                '''
            }
        }
        stage('Sintaxis') {
            steps { sh '.venv/bin/python -m compileall -q mlops_pipeline' }
        }
    }
    post {
        always {
            junit allowEmptyResults: true, testResults: 'junit.xml'
            archiveArtifacts allowEmptyArchive: true, artifacts: 'coverage.xml'
        }
        success { echo 'Jenkins CI: pruebas y estructura correctas.' }
        failure { echo 'Jenkins CI: error; revisar Console Output y reporte JUnit.' }
        cleanup {
            script {
                // Para avisos externos: definir CI_NOTIFICATION_EMAIL y configurar SMTP en Jenkins.
                if (env.CI_NOTIFICATION_EMAIL?.trim()) {
                    mail to: env.CI_NOTIFICATION_EMAIL,
                         subject: "MLOPS-CI ${currentBuild.currentResult}: ${env.BUILD_NUMBER}",
                         body: "Resultado: ${currentBuild.currentResult}\n${env.BUILD_URL}"
                } else {
                    echo 'Notificación por correo desactivada: falta CI_NOTIFICATION_EMAIL/SMTP.'
                }
            }
        }
    }
}
