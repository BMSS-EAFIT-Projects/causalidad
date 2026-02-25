import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

def calcular_propensity_score(df, tratamiento, covariables):
    """
    Estima el propensity score (P(Z=1 | X) mediante regresión logística.
    Retorna una copia del DataFrame con columna 'propensity_score' añadida.
    """
    df = df.dropna(subset=[tratamiento] + covariables).copy()
    if df[tratamiento].nunique() < 2:
        print("Error: No hay variación en el tratamiento.")
        return None
    formula = f"{tratamiento} ~ {' + '.join(covariables)}"
    try:
        modelo = smf.logit(formula, data=df).fit(disp=0)
        df['propensity_score'] = modelo.predict(df)
        return df
    except Exception as e:
        print(f"Error estimando propensity score: {e}")
        return None


def matched_sampling(df, tratamiento, ps_col='propensity_score', replacement=False, caliper=None):
    """
    Matching 1:1 por vecino más cercano.
    Retorna DataFrame con las filas emparejadas y columna 'pair_id'.
    """
    if 'T' not in df.columns:
        df = df.copy()
        df['T'] = df[tratamiento]
    tratados = df[df['T'] == 1]
    controles = df[df['T'] == 0]
    if tratados.empty or controles.empty:
        return None

    matched_t_idx = []
    matched_c_idx = []
    used_control = set() if not replacement else None

    if replacement:
        for t_idx in tratados.index:
            ps_t = tratados.loc[t_idx, ps_col]
            dist = abs(controles[ps_col] - ps_t)
            best = dist.idxmin()
            if caliper is None or dist[best] < caliper:
                matched_t_idx.append(t_idx)
                matched_c_idx.append(best)
    else:
        for t_idx in tratados.index:
            ps_t = tratados.loc[t_idx, ps_col]
            disponibles = controles[~controles.index.isin(used_control)]
            if disponibles.empty:
                break
            dist = abs(disponibles[ps_col] - ps_t)
            best = dist.idxmin()
            if caliper is None or dist[best] < caliper:
                matched_t_idx.append(t_idx)
                matched_c_idx.append(best)
                used_control.add(best)

    if not matched_t_idx:
        return None

    matched_t = df.loc[matched_t_idx].copy()
    matched_c = df.loc[matched_c_idx].copy()
    matched_t['pair_id'] = range(len(matched_t))
    matched_c['pair_id'] = range(len(matched_c))
    return pd.concat([matched_t, matched_c], ignore_index=True)


def subclassification(df, tratamiento, ps_col='propensity_score', n_subclases=5):
    """
    Divide la muestra en n_subclases según los quintiles del propensity score.
    Retorna DataFrame original con columnas 'subclase' y 'peso_subclase' añadidas.
    """
    df = df.copy()
    try:
        df['subclase'] = pd.qcut(df[ps_col], q=n_subclases, labels=False, duplicates='drop') + 1
    except ValueError:
        df['subclase'] = pd.cut(df[ps_col], bins=n_subclases, labels=False) + 1
    # Peso de cada subclase (proporción en la población)
    pesos = df['subclase'].value_counts(normalize=True).sort_index()
    df['peso_subclase'] = df['subclase'].map(pesos)
    return df


def balancear_propensity(df, tratamiento, resultado, covariables, metodo='matched',
                              n_subclases=5, replacement=False, caliper=None):
    """
    Aplica un método de balanceo basado en propensity score según el paper de Rosenbaum & Rubin.

    Parámetros:
    - df: DataFrame original.
    - tratamiento: nombre de la columna de tratamiento (valores 0/1).
    - resultado: nombre de la columna de resultado (solo para mantener interfaz, no se usa internamente).
    - covariables: lista de covariables para estimar el PS.
    - metodo: 'matched', 'subclassification' o 'covariance'.
    - n_subclases: número de subclases (para subclassification).
    - replacement: si se permite reemplazo en matching.
    - caliper: distancia máxima para matching.

    Retorna:
    - DataFrame transformado según el método elegido.
    - Si el método falla (ej. no se encuentra ningún match), retorna None.
    """
    # Primero, estimar propensity score
    df_ps = calcular_propensity_score(df, tratamiento, covariables)
    if df_ps is None:
        return None

    if metodo == 'matched':
        return matched_sampling(df_ps, tratamiento, ps_col='propensity_score',
                                replacement=replacement, caliper=caliper)
    elif metodo == 'subclassification':
        return subclassification(df_ps, tratamiento, ps_col='propensity_score',
                                 n_subclases=n_subclases)
    else:
        raise ValueError("Método no reconocido. Use 'matched', 'subclassification' o 'covariance'.")