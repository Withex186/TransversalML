import pandas as pd
import numpy as np
import os

# --- RUTAS DINAMICAS ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Entradas
FILE_APP = os.path.join(DATA_DIR, 'application_.parquet')
FILE_BUREAU = os.path.join(DATA_DIR, 'bureau.parquet')

# Salida (La tabla lista para entrenar)
FILE_OUT = os.path.join(DATA_DIR, 'integrated_master_table.parquet')

def run_integration():
    print("--- INICIANDO INTEGRACION DE FUENTES DE DATOS ---")
    
    # 1. Validar existencia de archivos
    if not os.path.exists(FILE_APP) or not os.path.exists(FILE_BUREAU):
        print(f" ERROR: Faltan archivos en {DATA_DIR}")
        return

    # 2. Cargar datasets
    print("⏳ Cargando parquets...")
    df_app = pd.read_parquet(FILE_APP)
    df_bureau = pd.read_parquet(FILE_BUREAU)
    
    print(f"   -> Application: {df_app.shape}")
    print(f"   -> Bureau:      {df_bureau.shape}")

    # 3. Ingeniería de caracteristicas agregadas (Requisito )
    # Aqui comprimimos el historial de N creditos pasados en una sola fila por cliente
    print("🔹 Generando variables agregadas del Buro...")
    
    buro_agg = df_bureau.groupby('SK_ID_CURR').agg({
        'DAYS_CREDIT': 'mean',        # ¿Hace cuanto pidio creditos en promedio?
        'AMT_CREDIT_SUM': 'sum',      # ¿Cuanto dinero le han prestado en total historicamente?
        'AMT_CREDIT_SUM_DEBT': 'sum'  # ¿Cuanto debe actualmente en total?
    }).reset_index()

    # Renombramos columnas para que sean claras
    buro_agg.columns = ['SK_ID_CURR', 'AVG_DAYS_CREDIT', 'TOTAL_PREV_LOAN_AMT', 'TOTAL_PREV_DEBT']

    # 4. Integracion (Merge) - Requisito 
    print("🔹 Uniendo tabla principal con datos del Buro...")
    # Usamos Left Join: Mantenemos todas las solicitudes, y pegamos info de buro si existe
    df_final = df_app.merge(buro_agg, on='SK_ID_CURR', how='left')

    # 5. Limpieza post-integración
    # Los clientes que NO cruzaron (no tienen historial en buro) tendran NaNs.
    # Logica de negocio: Si no tiene historial, su deuda previa es 0.
    new_cols = ['AVG_DAYS_CREDIT', 'TOTAL_PREV_LOAN_AMT', 'TOTAL_PREV_DEBT']
    df_final[new_cols] = df_final[new_cols].fillna(0)

    # 6. Guardar tabla maestra
    print(f"Guardando tabla integrada en: {FILE_OUT}")
    df_final.to_parquet(FILE_OUT)
    print(f"¡Éxito! Nueva dimensión del dataset: {df_final.shape}")

if __name__ == "__main__":
    run_integration()