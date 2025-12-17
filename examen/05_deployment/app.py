import pandas as pd
import joblib
import os
import sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- 1. CONFIGURACIÓN DE RUTAS E IMPORTACIONES ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, '02_data_preparation'))

try:
    from preprocessing import FeatureEngineer
except ImportError:
    raise ImportError("No se pudo importar FeatureEngineer.")

# --- 2. CARGAR RECURSOS ---
ARTIFACTS_DIR = os.path.join(parent_dir, 'artifacts')
DATA_DIR = os.path.join(parent_dir, 'data')
MODEL_FILE = os.path.join(ARTIFACTS_DIR, 'model.joblib')
X_TEST_FILE = os.path.join(DATA_DIR, 'X_test.parquet')

app = FastAPI(title="API de Riesgo de Crédito", version="1.0")

print("Cargando modelo...")
try:
    model_pipeline = joblib.load(MODEL_FILE)
    model_columns = pd.read_parquet(X_TEST_FILE).columns.tolist()
    print("Modelo cargado correctamente.")
except Exception as e:
    print(f"Error crítico: {e}")
    model_pipeline = None
    model_columns = []

# --- 3. FORMATO DE ENTRADA ---
class CreditApplication(BaseModel):
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    DAYS_BIRTH: int       
    DAYS_EMPLOYED: int    
    TOTAL_PREV_LOAN_AMT: float = 0.0
    TOTAL_PREV_DEBT: float = 0.0

# --- 4. ENDPOINT ---
@app.post("/evaluate_risk")
def predict_risk(application: CreditApplication):
    if not model_pipeline:
        raise HTTPException(status_code=500, detail="Modelo no cargado.")
    
    try:
        # A. Preparar datos
        input_data = pd.DataFrame([application.dict()])
        df_full = pd.DataFrame(columns=model_columns)
        df_predict = pd.concat([df_full, input_data], axis=0, ignore_index=True)
        df_predict = df_predict.fillna(0) # Relleno de seguridad
        
        # B. Predecir
        probability = model_pipeline.predict_proba(df_predict)[0][1]
        
        if probability < 0.44:
            decision = "APROBAR"
            recomendacion = "Cliente muy seguro. Credito pre-aprobado."
        elif probability >= 0.44 and probability < 0.49:
            decision = "REVISAR MANUALMENTE"
            recomendacion = "Riesgo moderado detectado. Se requiere validación superior."
        else:
            decision = "RECHAZAR"
            recomendacion = "Riesgo critico de impago. Rechazo automatico."
        
        return {
            "decision": decision,
            "riesgo_probabilidad": round(probability, 4),
            "mensaje": f"Probabilidad: {probability:.1%}. {recomendacion}"
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")