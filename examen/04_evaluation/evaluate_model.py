import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import sys
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# --- CONFIGURACION DE RUTAS ---
# Truco para importar modulos de carpetas hermanas
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, '02_data_preparation'))

# Importar la clase necesaria para cargar el pipeline
try:
    from preprocessing import FeatureEngineer
except ImportError:
    print("ERROR: No se pudo importar 'FeatureEngineer'. Verifica la carpeta 02.")
    sys.exit(1)

# Rutas
DATA_DIR = os.path.join(parent_dir, 'data')
ARTIFACTS_DIR = os.path.join(parent_dir, 'artifacts')
MODEL_FILE = os.path.join(ARTIFACTS_DIR, 'model.joblib')
X_TEST_FILE = os.path.join(DATA_DIR, 'X_test.parquet')
Y_TEST_FILE = os.path.join(DATA_DIR, 'y_test.parquet')

def evaluate():
    print("--- INICIANDO EVALUACIÓN DEL MODELO ---")

    # 1. Cargar Recursos
    if not os.path.exists(MODEL_FILE) or not os.path.exists(X_TEST_FILE):
        print("❌ Error: Faltan archivos (modelo o datos de test).")
        print("   -> Ejecuta primero el paso 03 (Entrenamiento).")
        return

    print("Loading pipeline & test data...")
    pipeline = joblib.load(MODEL_FILE)
    X_test = pd.read_parquet(X_TEST_FILE)
    y_test = pd.read_parquet(Y_TEST_FILE)
    
    # Si y_test es un DataFrame, extraemos la serie
    if isinstance(y_test, pd.DataFrame):
        y_test = y_test.iloc[:, 0]

    # 2. Generar predicciones
    print("Calculando predicciones...")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1] # Probabilidad de clase 1 (Incumplimiento)

    # 3. Metricas Numericas
    auc = roc_auc_score(y_test, y_prob)
    print(f"\n ROC-AUC Score: {auc:.4f}")
    print("   (Un valor > 0.7 es aceptable, > 0.8 es bueno)")
    
    print("\n Reporte de clasificacion:")
    print(classification_report(y_test, y_pred))

    # 4. Matriz de confusion (Grafico)
    print(" Generando grafico de matriz de confusion...")
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'Matriz de Confusion (AUC = {auc:.2f})')
    plt.xlabel('Prediccion (0: Paga, 1: Default)')
    plt.ylabel('Realidad')
    
    # Guardar evidencia
    output_img = os.path.join(ARTIFACTS_DIR, 'confusion_matrix.png')
    plt.savefig(output_img)
    plt.close()
    print(f" Grafico guardado en: {output_img}")

if __name__ == "__main__":
    evaluate()