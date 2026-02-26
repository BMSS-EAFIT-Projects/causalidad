import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

def visualizar_balance(df, tratamiento, covariables):
    """
    Retorna un DataFrame con el balance estadístico, define el balance
    basado en la prueba de igualdad de medias (p-valor > 0.05) y 
    muestra gráficos de densidad y boxplots.
    """
    df_clean = df[[tratamiento] + covariables].dropna().copy()
    
    tratados = df_clean[df_clean[tratamiento] == 1]
    controles = df_clean[df_clean[tratamiento] == 0]
    
    resultados = []
    
    n_covs = len(covariables)
    fig, axes = plt.subplots(n_covs, 2, figsize=(12, 4 * n_covs))
    
    if n_covs == 1:
        axes = [axes]
    
    for i, cov in enumerate(covariables):
        # Cálculos de medias y desviaciones
        mean_t = tratados[cov].mean()
        std_t = tratados[cov].std()
        
        mean_c = controles[cov].mean()
        std_c = controles[cov].std()
        
        # Prueba T de igualdad de medias
        t_stat, p_val = ttest_ind(tratados[cov], controles[cov], equal_var=False)
        
        # SMD como métrica complementaria
        var_pooled = (tratados[cov].var() + controles[cov].var()) / 2
        std_pooled = np.sqrt(var_pooled)
        smd = abs((mean_t - mean_c) / std_pooled) if std_pooled != 0 else 0
        
        # --- NUEVO CRITERIO BASADO EN IGUALDAD DE MEDIAS ---
        es_balanceado = "Sí" if p_val > 0.05 else "No"
        
        resultados.append({
            'Covariable': cov,
            'Tratados (Media ± SD)': f"{mean_t:.2f} ± {std_t:.2f}",
            'Controles (Media ± SD)': f"{mean_c:.2f} ± {std_c:.2f}",
            'p-valor (Prueba de Medias)': round(p_val, 4),
            'SMD': round(smd, 4), # El SMD es e
            'Balanceado': es_balanceado
        })
        
        # Gráficos visuales para confirmar que "casi iguales" se vea en la forma de la campana
        sns.kdeplot(data=df_clean, x=cov, hue=tratamiento, fill=True, 
                    common_norm=False, alpha=0.5, ax=axes[i][0])
        axes[i][0].set_title(f'Distribución de Densidad: {cov}')
        
        sns.boxplot(data=df_clean, x=tratamiento, y=cov, ax=axes[i][1])
        axes[i][1].set_title(f'Boxplot: {cov}')
        
    plt.tight_layout()
    plt.show()
    
    return pd.DataFrame(resultados)