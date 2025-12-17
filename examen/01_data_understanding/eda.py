import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIGURACIÓN DE RUTAS ---
# Subimos un nivel desde '01_data_understanding' para llegar a la raiz
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')

# Archivos requeridos
FILE_APP = os.path.join(DATA_DIR, 'application_.parquet')
FILE_BUREAU = os.path.join(DATA_DIR, 'bureau.parquet')

def analizar_dataset(df, nombre):
    "Función auxiliar para no repetir codigo"
    print(f"\n--- ANALISIS DE: {nombre.upper()} ---")
    print(f" Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
    
    print(f" Nulos (Top 5):")
    missing = (df.isnull().sum() / len(df)) * 100
    print(missing.sort_values(ascending=False).head(5))
    
    # Revisar si tiene la llave para el cruce (SK_ID_CURR)
    if 'SK_ID_CURR' in df.columns:
        unique_ids = df['SK_ID_CURR'].nunique()
        print(f"🔑 IDs unicos (SK_ID_CURR): {unique_ids}")
    else:
        print(" ADVERTENCIA: Este dataset no tiene la columna 'SK_ID_CURR'.")

def run_eda():
    print("--- INICIANDO ANALISIS EXPLORATORIO (EDA) COMPLETO ---")
    
    # 1. Verificar archivos
    if not os.path.exists(FILE_APP) or not os.path.exists(FILE_BUREAU):
        print(f" ERROR: Faltan archivos en {DATA_DIR}")
        print("   -> Asegúrate de tener 'application.parquet' Y 'bureau.parquet'.")
        return

    # 2. Cargar Datos
    print(" Cargando datos (esto puede tardar)...")
    try:
        df_app = pd.read_parquet(FILE_APP)
        df_bureau = pd.read_parquet(FILE_BUREAU)
    except Exception as e:
        print(f" Error leyendo parquets: {e}")
        return

    # 3. Analizar application (Tabla Principal)
    analizar_dataset(df_app, "Application (Solicitudes)")

    # Analisis especifico del TARGET (Requisito clave: Desbalance)
    if 'TARGET' in df_app.columns:
        print("\n🎯 Distribucion del TARGET (Riesgo):")
        print(df_app['TARGET'].value_counts(normalize=True) * 100)
        
        # Gráfico de Evidencia
        os.makedirs(ARTIFACTS_DIR, exist_ok=True)
        plt.figure(figsize=(6, 4))
        sns.countplot(x=df_app['TARGET'], palette='viridis')
        plt.title('Evidencia: Desbalance de clases (Application)')
        output_img = os.path.join(ARTIFACTS_DIR, 'target_distribution.png')
        plt.savefig(output_img)
        print(f"✅ Grafico guardado en: {output_img}")

    # 4. Analizar Bureau (Tabla Secundaria)
    analizar_dataset(df_bureau, "Bureau (Historial Crédito)")

    # 5. Validación de integración (Adelantando el requisito de integración)
    print("\n--- 🔗 ANALISIS DE INTEGRACIÓN ---")
    ids_app = set(df_app['SK_ID_CURR'])
    ids_bureau = set(df_bureau['SK_ID_CURR'])
    
    coincidencias = ids_app.intersection(ids_bureau)
    print(f"Clientes en Application: {len(ids_app)}")
    print(f"Clientes con historial en Bureau: {len(ids_bureau)}")
    print(f"Clientes coincidentes (Crucables): {len(coincidencias)}")
    print(f"   -> {len(coincidencias) / len(ids_app):.1%} de las solicitudes tienen datos en buro.")

if __name__ == "__main__":
    run_eda()