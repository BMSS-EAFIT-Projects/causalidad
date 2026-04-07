"""
propensity_score.py
===================
Estima pi(X) = P(A=1|X) y lo agrega como columna al DataFrame.

Uso
---
    from propensity_score import propensity_score

    df = propensity_score(df, tratamiento='treatment', covariables=covs)
    # df ahora tiene columna 'propensity_score'
"""

import numpy as np
import statsmodels.formula.api as smf
import warnings


def propensity_score(df, tratamiento, covariables,
                     columna='propensity_score'):
    """
    Estima pi_hat(X) = P(A=1|X) con regresión logística y lo guarda
    como columna en el DataFrame.

    Equivalente en R (slides de LaLonde / Taddeo 2025, p.68 y p.112):
        PS_fit <- glm(treatment ~ ., data=df, family="binomial")
        df$propensity_score <- predict(PS_fit, type="response")

    Parámetros
    ----------
    df          : pandas DataFrame original.
    tratamiento : Nombre de la columna de tratamiento (valores 0/1).
    covariables : Lista de covariables que entran en el modelo logístico.
    columna     : Nombre de la columna donde se guarda el PS
                  (default 'propensity_score').

    Retorna
    -------
    pandas DataFrame idéntico al original con la columna del PS añadida.
    Rango del PS garantizado: (0, 1) estrictamente — se aplica clip
    numérico (1e-8) para evitar divisiones por cero en los IPW.

    Raises
    ------
    ValueError : Si el tratamiento no tiene variación.
    ValueError : Si alguna covariable no existe en df.

    Notas
    -----
    · Si quieres un modelo más rico (como en los slides p.112 donde se
      usan I(education^2) e I(education^3)), incluye esos términos
      directamente en la lista de covariables:
          covs = ['age', 'education', 'education**2', 'education**3', ...]

    · Si la columna ya existe en df, se sobreescribe con un aviso.

    Ejemplo
    -------
    df = propensity_score(
            df          = lalonde,
            tratamiento = 'treatment',
            covariables = ['age', 'education', 'black',
                           'hispanic', 'married', 'nodegree',
                           're74', 're75']
         )
    # Usar columna en balance():
    df_bal = balance(df, tratamiento='treatment', metodo='trimming')
    """
    # ── Validaciones ──────────────────────────────────────────
    cols_faltantes = [c for c in [tratamiento] + covariables
                      if c not in df.columns]
    if cols_faltantes:
        raise ValueError(f"Columnas no encontradas en df: {cols_faltantes}")

    if df[tratamiento].nunique() < 2:
        raise ValueError(
            "El tratamiento no tiene variación — "
            "todos los valores son iguales."
        )

    if columna in df.columns:
        print(f"Aviso: la columna '{columna}' ya existe y será sobreescrita.")

    # ── Estimación ────────────────────────────────────────────
    df_clean = df[[tratamiento] + covariables].dropna()
    formula  = f"{tratamiento} ~ {' + '.join(covariables)}"

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            # Intento 1: Newton (método por defecto)
            modelo = smf.logit(formula, data=df_clean).fit(disp=0)
        except np.linalg.LinAlgError:
            try:
                # Intento 2: BFGS — no requiere invertir la Hessiana
                modelo = smf.logit(formula, data=df_clean).fit(
                    method='bfgs', disp=0
                )
                print("  Info: Newton falló (matriz singular), se usó BFGS.")
            except Exception:
                # Intento 3: Ridge logístico (L2) — maneja multicolinealidad perfecta
                modelo = smf.logit(formula, data=df_clean).fit_regularized(
                    method='l1', alpha=0.1, disp=0
                )
                print("  Info: BFGS falló, se usó regresión logística regularizada (L1, alpha=0.1).")

    ps = modelo.predict(df_clean).values
    ps = np.clip(ps, 1e-8, 1 - 1e-8)

    # Guardar en el df original (NaN para filas que se cayeron por dropna)
    df = df.copy()
    df[columna] = np.nan
    df.loc[df_clean.index, columna] = ps

    # ── Diagnóstico ───────────────────────────────────────────
    n_t = int((df_clean[tratamiento] == 1).sum())
    n_c = int((df_clean[tratamiento] == 0).sum())

    print(f"propensity_score() — n={len(df_clean)} (T={n_t}, C={n_c})")
    print(f"  PS: min={ps.min():.4f}  p25={np.percentile(ps,25):.4f}  "
          f"media={ps.mean():.4f}  p75={np.percentile(ps,75):.4f}  "
          f"max={ps.max():.4f}")

    n_extremos = ((ps < 0.05) | (ps > 0.95)).sum()
    if n_extremos > 0:
        pct = n_extremos / len(ps) * 100
        print(f"  Aviso: {n_extremos} unidades ({pct:.1f}%) con PS < 0.05 "
              f"o PS > 0.95 — considera trimming o truncating.")

    return df