import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

def visualizar_balance(df, tratamiento, covariables):
    """
    Retorna un DataFrame con el balance estadístico y muestra en pantalla 
    los gráficos de densidad y boxplots de las covariables entre grupos.
    """
    # Filtrar datos sin nulos
    df_clean = df[[tratamiento] + covariables].dropna().copy()
    
    tratados = df_clean[df_clean[tratamiento] == 1]
    controles = df_clean[df_clean[tratamiento] == 0]
    
    resultados = []
    
    # Configurar la cuadrícula de gráficos (1 fila por covariable, 2 columnas)
    n_covs = len(covariables)
    fig, axes = plt.subplots(n_covs, 2, figsize=(12, 4 * n_covs))
    
    # Asegurar que axes sea iterable bidimensional incluso si hay solo 1 covariable
    if n_covs == 1:
        axes = [axes]
    
    for i, cov in enumerate(covariables):
        # --- 1. CÁLCULOS ESTADÍSTICOS ---
        mean_t = tratados[cov].mean()
        std_t = tratados[cov].std()
        
        mean_c = controles[cov].mean()
        std_c = controles[cov].std()
        
        # Prueba T y SMD
        t_stat, p_val = ttest_ind(tratados[cov], controles[cov], equal_var=False)
        var_pooled = (tratados[cov].var() + controles[cov].var()) / 2
        std_pooled = np.sqrt(var_pooled)
        smd = (mean_t - mean_c) / std_pooled if std_pooled != 0 else 0
        
        resultados.append({
            'Covariable': cov,
            'Tratados (Media ± SD)': f"{mean_t:.2f} ± {std_t:.2f}",
            'Controles (Media ± SD)': f"{mean_c:.2f} ± {std_c:.2f}",
            'p-valor': round(p_val, 4),
            'SMD': round(abs(smd), 4)
        })
        
        # --- 2. GRÁFICO DE DENSIDAD ---
        sns.kdeplot(data=df_clean, x=cov, hue=tratamiento, fill=True, 
                    common_norm=False, alpha=0.5, ax=axes[i][0])
        axes[i][0].set_title(f'Distribución de Densidad: {cov}')
        
        # --- 3. BOXPLOT ---
        sns.boxplot(data=df_clean, x=tratamiento, y=cov, ax=axes[i][1])
        axes[i][1].set_title(f'Boxplot: {cov}')
        
    # Ajustar diseño y mostrar la imagen en pantalla (sin guardar)
    plt.tight_layout()
    plt.show()
    
    # Convertir resultados a DataFrame y retornarlo para que el entorno lo muestre una sola vez
    df_balance = pd.DataFrame(resultados)
    
    return df_balance