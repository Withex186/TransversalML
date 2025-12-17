import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Clase personalizada para el Pipeline. 
# Es crucial para que el modelo pueda limpiar datos nuevos automaticamente en la API.
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        
        # 1. Transformacion de EDAD (DAYS_BIRTH viene negativo en el original)
        if 'DAYS_BIRTH' in X_copy.columns:
            # Convertimos a años positivos
            X_copy['DAYS_BIRTH'] = X_copy['DAYS_BIRTH'] / -365.0
        
        # 2. Limpieza de dias empleado (Valor 365243 es un error conocido en este dataset)
        if 'DAYS_EMPLOYED' in X_copy.columns:
            X_copy['DAYS_EMPLOYED'] = X_copy['DAYS_EMPLOYED'].replace(365243, np.nan)
            # Convertimos a años positivos
            X_copy['DAYS_EMPLOYED'] = X_copy['DAYS_EMPLOYED'] / -365.0
            
        return X_copy