import pandas as pd
import statsmodels.formula.api as smf

def calcular_ate(df, resultado, tratamiento, covariables):
    """
    Calcula el ATE (Average Treatment Effect) para un resultado binario
    y un tratamiento binario, utilizando diferencia de medias y regresión
    lineal con covariables.

    Parámetros
    ----------
    df : pandas.DataFrame
        DataFrame que contiene las variables.
    resultado : str
        Nombre de la columna de resultado (se espera binaria 0/1).
    tratamiento : str
        Nombre de la columna de tratamiento (se espera binaria 0/1).
    covariables : list of str
        Lista de nombres de covariables (numéricas).

    Retorna
    -------
    dict
        Diccionario con las siguientes claves:
        - 'ate_diff' : float - Diferencia de medias (tratados - control)
        - 'ate_reg' : float - Coeficiente de tratamiento en regresión lineal con covariables
        - 'p_valor_reg' : float - P-valor de ese coeficiente
        - 'n_tratados' : int - Número de unidades con tratamiento == 1
        - 'n_controles' : int - Número de unidades con tratamiento == 0

        Si no es posible estimar (falta variación o datos), retorna None.
    """
    # Seleccionar columnas necesarias y eliminar filas con NaN
    columnas = [resultado, tratamiento] + covariables
    df_clean = df[columnas].dropna()

    if df_clean.empty:
        print("Error: No hay datos completos para las variables especificadas.")
        return None

    # Identificar grupos de tratamiento y control (asumiendo valores 0/1)
    tratados = df_clean[tratamiento] == 1
    controles = df_clean[tratamiento] == 0

    n_tratados = tratados.sum()
    n_controles = controles.sum()

    if n_tratados == 0 or n_controles == 0:
        print("Error: No hay variación en el tratamiento (todos son 0 o todos son 1).")
        return None

    # ATE por diferencia de medias
    ate_diff = df_clean.loc[tratados, resultado].mean() - df_clean.loc[controles, resultado].mean()

    # ATE por regresión lineal con covariables
    formula = f"{resultado} ~ {tratamiento} + {' + '.join(covariables)}"
    modelo = smf.ols(formula, data=df_clean).fit()
    ate_reg = modelo.params[tratamiento]
    p_valor_reg = modelo.pvalues[tratamiento]

    return {
        'ate_diff': ate_diff,
        'ate_reg': ate_reg,
        'p_valor_reg': p_valor_reg,
        'n_tratados': int(n_tratados),
        'n_controles': int(n_controles)
    }