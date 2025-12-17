Predicción de Riesgo de Crédito (Home Credit Default Risk)

Este proyecto implementa una solución de Machine Learning end-to-end para predecir la probabilidad de incumplimiento de pago de clientes bancarios.
El sistema sigue la metodología CRISP-DM y una arquitectura de microservicios.

Estructura del Proyecto

El codigo esta modularizado siguiendo los requisitos del examen:

/01_data_understanding**: Análisis Exploratorio de Datos (EDA) y validación de fuentes.
/02_data_preparation**: Ingeniería de características e integración de tablas (`application` + `bureau`).
/03_modeling**: Entrenamiento del modelo RandomForest y pipeline de preprocesamiento.
/04_evaluation**: Validación del modelo con métricas (ROC-AUC) y matrices de confusión.
/05_deployment**: API REST construida con **FastAPI** para predicciones en tiempo real.
/artifacts**: Almacenamiento de modelos serializados (`.joblib`) y gráficos de reporte.

Instrucciones de Ejecucion

1. Instalación
Instalar las dependencias necesarias para la ejecucion de los scripts:
pip install -r requirements.txt

2. Ejecucion de scripts:
2.1 python 01_data_understanding/eda.py # Con este comando podemos ejecutar el primer scripts,
                                        # con el cual cargara los archivos application_.parquet y el bureau.parquet para inspeccionar su estructura, verificar la calidad de los datos 
                                        # y confirmar que las tablas se pueden cruzar.

2.2 python 02_data_preparation/data_integration.py  # Con este comando se ejecuta el segundo scripts, donde usara la base de "application_.parquet" par tomar la tabla "datos del cliente"
                                                    # y de "bureau.parquet" tomara la tabla "historial externo", para agregar datos del buro por cliente, creando la tabla 
                                                    # "integrated_master_table.parquet", lista para empezar a moderlar.

2.3 python 03_modeling/train.py # Con este comando se ejecutara el tercer scripts, para poder leer la tabla que se genero con el segundo script, construyendo un flujo de tranformacion
                                # automatizado, lo que incluye: Limpieza personalizada importada de la carpeta "02_data_preparation" usando el archivo "preprocessing.py"
                                # para rellenar valores nulos con la media y normalizacion de datos numericos.
                                # Tambien se entrena el modelo con RandomForestClassifier optimizado para clases desbalanceadas, guardando el modelo ¨.joblib¨ entrenado en la carpeta data
                                # con el nombre "X_test" y "Y_test".

2.4 python 04_evaluation/evaluate_model.py # Al ejecutar este comando cargara el modelo guardado y los datos de prueba, calculando el ROC-AUC Score y genera un reporte de clasificacion
                                            # Creando la carpeta "artifacts" para guardar "confusion_matrix.png" y "target_distribution.png" como evidencia grafica.

2.5 uvicorn 05_deployment.app:app --reload  # Para el ultimo script se ejecutara la API, disponible en "http://127.0.0.1:8000/docs" la cual se utilizo FastAPI para ingresar datos del cliente
                                            # y ver si se le aprueba o rechaza el credito mostrando tambien el riesgo de probabilidad de que no pague.