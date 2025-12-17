import pandas as pd
import numpy as np
import os
import sys
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# --- CONFIGURACIÓN DE RUTAS ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, '02_data_preparation'))

try:
    from preprocessing import FeatureEngineer
except ImportError:
    print("ERROR CRÍTICO: No se pudo importar 'FeatureEngineer'.")
    sys.exit(1)

DATA_DIR = os.path.join(parent_dir, 'data')
ARTIFACTS_DIR = os.path.join(parent_dir, 'artifacts')
INPUT_FILE = os.path.join(DATA_DIR, 'integrated_master_table.parquet')
MODEL_FILE = os.path.join(ARTIFACTS_DIR, 'model.joblib')

def train_model():
    print("--- INICIANDO ENTRENAMIENTO DEL MODELO ---")

    # 1. Cargar Datos
    if not os.path.exists(INPUT_FILE):
        print(f"Error: No se encontró {INPUT_FILE}")
        return
    else:
        print(f"Cargando tabla maestra: {INPUT_FILE}")
        df = pd.read_parquet(INPUT_FILE)

    # 2. Separar Features (X) y Target (y)
    if 'TARGET' not in df.columns:
        print("Error: La columna 'TARGET' no está en el dataset.")
        return

    # Eliminamos ID y Target
    X = df.drop(columns=['TARGET', 'SK_ID_CURR'])
    y = df['TARGET']

    # --- CORRECCIÓN AQUÍ: FILTRO NUMÉRICO ---
    # Seleccionamos solo columnas numéricas para evitar errores con textos como 'Cash loans'
    # Esto soluciona el error del SimpleImputer(strategy='median')
    X = X.select_dtypes(include=[np.number])
    
    print(f"ℹ️ Entrenando con {X.shape[1]} variables numéricas.")

    # 3. Split Train/Test
    print("Dividiendo datos (Train/Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Guardamos los datos de test para usarlos en el paso 4 (Evaluación)
    # Esto es vital para que evaluate_model.py funcione después
    X_test.to_parquet(os.path.join(DATA_DIR, 'X_test.parquet'))
    pd.DataFrame(y_test).to_parquet(os.path.join(DATA_DIR, 'y_test.parquet'))
    print("✅ Sets de prueba guardados en la carpeta data.")

    # 4. Definir Pipeline
    pipeline = Pipeline([
        ('ingenieria', FeatureEngineer()),            
        ('imputador', SimpleImputer(strategy='median')), 
        ('escalador', StandardScaler()),              
        ('modelo', RandomForestClassifier(            
            n_estimators=100, 
            max_depth=10, 
            class_weight='balanced', 
            random_state=42,
            n_jobs=-1
        ))
    ])

    # 5. Entrenar
    print("Entrenando (esto puede tardar un poco)...")
    try:
        pipeline.fit(X_train, y_train)
    except Exception as e:
        print(f"Error durante el entrenamiento: {e}")
        return

    print("Entrenamiento finalizado.")

    # 6. Guardar Modelo
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_FILE)
    print(f"\nModelo guardado exitosamente en: {MODEL_FILE}")

if __name__ == "__main__":
    train_model()