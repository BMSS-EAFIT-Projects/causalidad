"""
calcular_ate.py
===============
Estimadores del ATE fiel a las diapositivas de LaLonde (Taddeo, 2025).

Recibe el DataFrame ya balanceado (output de balance()) con la columna
'propensity_score' incluida. No estima el PS internamente.

Estimadores
-----------
1. naive          — diferencia de medias E[Y|A=1] - E[Y|A=0]          [slide 5/9]
2. regresion      — OLS: Y ~ A + X, coeficiente de A                   [slide 5/9]
3. g_formula      — estandarización: promedio de Ŷ(1) - Ŷ(0)          [slide 87]
4. ht             — Horvitz-Thompson (IPW no normalizado)               [slide 54]
5. hajek          — Hajek (IPW normalizado)                             [slide 65]
6. msm            — Modelo Estructural Marginal (WLS pesos estabilizados)[slide 77]
7. dr             — Estimador Doblemente Robusto (AIPW)                 [slide 78]

Uso
---
    from calcular_ate import calcular_ate

    resultados = calcular_ate(
        df          = df_balanceado,
        resultado   = 'ingreso',
        tratamiento = 'capacitacion',
        covariables = ['edad', 'educacion', 'raza'],
        col_ps      = 'propensity_score'   # columna generada por propensity_score()
    )
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


# ══════════════════════════════════════════════════════════════
# 1. NAIVE — diferencia de medias
# ══════════════════════════════════════════════════════════════

def naive(df, resultado, tratamiento):
    """
    Estimador naïve: E[Y|A=1] - E[Y|A=0].

    Slide 5 (datos randomizados):
        mean(lalonde$re78[lalonde$treat==1]) - mean(lalonde$re78[lalonde$treat==0])

    Solo es válido bajo asignación aleatoria. En datos observacionales
    confunde el efecto causal con sesgo de selección.

    Retorna
    -------
    float : diferencia de medias.
    """
    y1 = df.loc[df[tratamiento] == 1, resultado]
    y0 = df.loc[df[tratamiento] == 0, resultado]
    return float(y1.mean() - y0.mean())


# ══════════════════════════════════════════════════════════════
# 2. REGRESIÓN LINEAL — OLS condicional
# ══════════════════════════════════════════════════════════════

def regresion(df, resultado, tratamiento, covariables):
    """
    OLS condicional: Y ~ A + X₁ + X₂ + ...

    Slide 5 (datos randomizados):
        lalonde.fit <- lm(re78 ~ ., data=lalonde)
        coef['treat']

    Slide 9 (datos observacionales con CPS):
        lalonde_fit <- lm(re78 ~ ., data=lalonde)
        coef['treatment']

    El coeficiente de A es el ATE bajo CIA y correcta especificación lineal.

    Retorna
    -------
    dict con 'ate', 'ic_95', 'p_valor'.
    """
    formula = f"{resultado} ~ {tratamiento} + {' + '.join(covariables)}"
    modelo  = smf.ols(formula, data=df).fit()
    ic      = modelo.conf_int().loc[tratamiento]

    return {
        'ate'    : float(modelo.params[tratamiento]),
        'ic_95'  : (float(ic[0]), float(ic[1])),
        'p_valor': float(modelo.pvalues[tratamiento]),
    }


# ══════════════════════════════════════════════════════════════
# 3. G-FÓRMULA — estandarización
# ══════════════════════════════════════════════════════════════

def g_formula(df, resultado, tratamiento, covariables):
    """
    G-fórmula (outcome regression / estandarización).

    Slide 87 (ejemplo LaLonde ATT via g-formula):
        lalonde_fit <- lm(re78 ~ treatment + age + education + ..., data=lalonde)
        TEMP <- lalonde
        TEMP$treatment <- 0
        lalonde_pred <- predict(lalonde_fit, newdata=TEMP)
        ATT = mean(re78[treat==1]) - mean(lalonde_pred[treat==1])

    Para ATE se impone A=1 y A=0 sobre TODA la muestra:
        ATE = (1/n) Σ [Ŷᵢ(1) - Ŷᵢ(0)]

    Modelo usado según tipo de resultado
    -------------------------------------
    · Continuo  → OLS  (lm en R)
      En este caso g-fórmula = regresión OLS algebraicamente.
      Son idénticos porque Ŷ(1)-Ŷ(0) = β₁ para toda i.

    · Binario (0/1) → Logit  (glm family=binomial en R)
      Aquí sí difieren: las predicciones son probabilidades no lineales
      en A, por lo que Ŷᵢ(1)-Ŷᵢ(0) varía entre individuos y el promedio
      ya no colapsa al coeficiente de regresión.

    Detección automática: si el resultado tiene exactamente dos valores
    únicos (0 y 1), se usa logit. En cualquier otro caso, OLS.

    Retorna
    -------
    dict con 'ate', 'modelo_usado' y 'efectos_individuales'.
    """
    valores_unicos = df[resultado].dropna().unique()
    es_binario     = set(valores_unicos).issubset({0, 1})

    formula = f"{resultado} ~ {tratamiento} + {' + '.join(covariables)}"

    if es_binario:
        try:
            modelo = smf.logit(formula, data=df).fit(disp=0)
        except np.linalg.LinAlgError:
            try:
                modelo = smf.logit(formula, data=df).fit(method='bfgs', disp=0)
            except Exception:
                modelo = smf.logit(formula, data=df).fit_regularized(
                    method='l1', alpha=0.1, disp=0
                )
        modelo_used = 'logit'
    else:
        modelo      = smf.ols(formula, data=df).fit()
        modelo_used = 'ols'

    df1 = df.copy(); df1[tratamiento] = 1
    df0 = df.copy(); df0[tratamiento] = 0

    y1_hat = modelo.predict(df1).values
    y0_hat = modelo.predict(df0).values
    efecto = y1_hat - y0_hat

    return {
        'ate'                 : float(efecto.mean()),
        'modelo_usado'        : modelo_used,
        'efectos_individuales': pd.Series(efecto, index=df.index),
    }


# ══════════════════════════════════════════════════════════════
# 4. HORVITZ-THOMPSON — IPW no normalizado
# ══════════════════════════════════════════════════════════════

def ht(df, resultado, tratamiento, col_ps):
    """
    Estimador de Horvitz-Thompson.

    Slide 54 — fórmula exacta:
        ATE_HT = (1/n) Σ AᵢYᵢ/π(Xᵢ)  -  (1/n) Σ (1-Aᵢ)Yᵢ/(1-π(Xᵢ))

    Slide 102 (ejemplo LaLonde):
        with(DATA, mean(School_meal * BMI * IPW) - mean((1-School_meal) * BMI * IPW))

    donde IPW = A/π + (1-A)/(1-π), que expandido da exactamente la fórmula de arriba.

    Sensible a PS extremos (slide 64: "HT will fail if π(Xᵢ) ≈ 0 or ≈ 1").

    Retorna
    -------
    float : ATE estimado.
    """
    A  = df[tratamiento].values.astype(float)
    Y  = df[resultado].values.astype(float)
    ps = df[col_ps].values.astype(float)

    return float(np.mean(A * Y / ps) - np.mean((1 - A) * Y / (1 - ps)))


# ══════════════════════════════════════════════════════════════
# 5. HAJEK — IPW normalizado
# ══════════════════════════════════════════════════════════════

def hajek(df, resultado, tratamiento, col_ps):
    """
    Estimador de Hajek (IPW normalizado / self-normalized).

    Slide 65 — fórmula exacta:
        ATE_Hajek = [Σ AᵢYᵢ/π(Xᵢ)] / [Σ Aᵢ/π(Xᵢ)]
                  - [Σ (1-Aᵢ)Yᵢ/(1-π(Xᵢ))] / [Σ (1-Aᵢ)/(1-π(Xᵢ))]

    Slide 115 (ejemplo LaLonde):
        (sum(treatment * re78 * IPW) / sum(treatment * IPW)) -
        (sum((1-treatment) * re78 * IPW) / sum((1-treatment) * IPW))

    Normalizar elimina la falta de invarianza del HT ante traslaciones de Y
    (slide 64-65: "Normalizing the weights removes the lack of invariance").

    Retorna
    -------
    float : ATE estimado.
    """
    A  = df[tratamiento].values.astype(float)
    Y  = df[resultado].values.astype(float)
    ps = df[col_ps].values.astype(float)

    mu1 = np.sum(A * Y / ps)       / np.sum(A / ps)
    mu0 = np.sum((1-A) * Y / (1-ps)) / np.sum((1-A) / (1-ps))

    return float(mu1 - mu0)


# ══════════════════════════════════════════════════════════════
# 6. MSM — Modelo Estructural Marginal
# ══════════════════════════════════════════════════════════════

def msm(df, resultado, tratamiento, col_ps):
    """
    Modelo Estructural Marginal (WLS con pesos estabilizados).

    Slide 77 — modelo y pesos exactos:
        Suponer E[Y(a)] = β₀ + β₁·a  =>  ATE = β₁

        Estimar (β̂₀, β̂₁) = argmin Σ W̃ᵢ (Yᵢ - α₀ - α₁·Aᵢ)²

        donde los pesos estabilizados son:
            W̃ᵢ = E(A) / π(Xᵢ)

    Slide 77:
        "W_tilde_i = E(A) / π(Xᵢ)   [STABILIZED WEIGHTS]"

    El modelo marginal es solo Y ~ A (sin covariables), porque el MSM
    modela la media marginal del contrafactual E[Y(a)], no la condicional.
    Los pesos se encargan de eliminar la confusión.

    Retorna
    -------
    dict con 'ate', 'ic_95', 'p_valor'.
    """
    A  = df[tratamiento].values.astype(float)
    ps = df[col_ps].values.astype(float)

    # Pesos estabilizados: W̃ᵢ = E(A) / π(Xᵢ)  [slide 77]
    e_a   = A.mean()
    pesos = np.where(A == 1, e_a / ps, (1 - e_a) / (1 - ps))

    df_msm        = df[[resultado, tratamiento]].copy()
    df_msm['_w']  = pesos

    formula = f"{resultado} ~ {tratamiento}"
    modelo  = smf.wls(formula, data=df_msm, weights=df_msm['_w']).fit()
    ic      = modelo.conf_int().loc[tratamiento]

    return {
        'ate'    : float(modelo.params[tratamiento]),
        'ic_95'  : (float(ic[0]), float(ic[1])),
        'p_valor': float(modelo.pvalues[tratamiento]),
    }


# ══════════════════════════════════════════════════════════════
# 7. DR — Estimador Doblemente Robusto
# ══════════════════════════════════════════════════════════════

def dr(df, resultado, tratamiento, covariables, col_ps):
    """
    Estimador Doblemente Robusto (AIPW).

    Slide 78 — forma equivalente (la más directa para implementar):

        m₁ = E[ A(Y - μ₁(X)) / π(X)  +  μ₁(X) ]
        m₀ = E[ (1-A)(Y - μ₀(X)) / (1-π(X))  +  μ₀(X) ]
        ATE = m₁ - m₀

    donde μₐ(X) = E[Y|A=a, X]  (modelo de resultado, estimado con OLS).

    Slide 79 — propiedad de doble robustez:
        "ATE = m₁ - m₀  si  π(X;θ) o (μ₀, μ₁) está correctamente especificado"

    Es consistente si al menos uno de los dos modelos (PS o resultado)
    está bien especificado. Si ambos lo están, es semiparamétricamente eficiente.

    Retorna
    -------
    dict con 'ate'.
    """
    # ── Modelo de resultado: μₐ(X) = E[Y|A=a, X] ─────────────
    formula = f"{resultado} ~ {tratamiento} + {' + '.join(covariables)}"
    modelo  = smf.ols(formula, data=df).fit()

    df1 = df.copy(); df1[tratamiento] = 1
    df0 = df.copy(); df0[tratamiento] = 0

    mu1 = modelo.predict(df1).values   # μ₁(Xᵢ)
    mu0 = modelo.predict(df0).values   # μ₀(Xᵢ)

    A  = df[tratamiento].values.astype(float)
    Y  = df[resultado].values.astype(float)
    ps = df[col_ps].values.astype(float)

    # ── Forma equivalente slide 78 ────────────────────────────
    # m₁ᵢ = A(Y - μ₁) / π  +  μ₁
    # m₀ᵢ = (1-A)(Y - μ₀) / (1-π)  +  μ₀
    m1 = A * (Y - mu1) / ps      + mu1
    m0 = (1 - A) * (Y - mu0) / (1 - ps) + mu0

    return {
        'ate': float(np.mean(m1) - np.mean(m0)),
    }


# ══════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════

def calcular_ate(df, resultado, tratamiento, covariables,
                 col_ps='propensity_score'):
    """
    Calcula el ATE con todos los estimadores disponibles.

    Parámetros
    ----------
    df          : DataFrame balanceado con columna del PS ya calculada.
                  Output de balance() o el df original con propensity_score().
    resultado   : Nombre de la columna de resultado.
    tratamiento : Nombre de la columna de tratamiento (0/1).
    covariables : Lista de covariables (para regresión, g-fórmula y DR).
    col_ps      : Nombre de la columna del PS (default 'propensity_score').
                  Para truncating usar 'propensity_score' (ya está recortado).

    Retorna
    -------
    dict con los ATE de todos los estimadores. Las claves con '_ic95'
    contienen tuplas (inf, sup) y las claves con '_pvalor' el p-valor.
    Retorna None si los datos son inválidos.

    Ejemplo
    -------
    res = calcular_ate(df_trim, 're78', 'treatment', covs)
    print(f"DR:    {res['dr']:.2f}")
    print(f"Hajek: {res['hajek']:.2f}")
    """
    # ── Validaciones ──────────────────────────────────────────
    if col_ps not in df.columns:
        raise ValueError(
            f"La columna '{col_ps}' no existe. "
            "Llama primero a propensity_score()."
        )

    cols_req = [resultado, tratamiento] + covariables
    faltantes = [c for c in cols_req if c not in df.columns]
    if faltantes:
        raise ValueError(f"Columnas no encontradas en df: {faltantes}")

    df_clean = df.dropna(subset=cols_req + [col_ps]).copy()

    if df_clean.empty:
        print("calcular_ate() — Error: sin datos completos.")
        return None

    n_t = int((df_clean[tratamiento] == 1).sum())
    n_c = int((df_clean[tratamiento] == 0).sum())

    if n_t == 0 or n_c == 0:
        print("calcular_ate() — Error: sin variación en el tratamiento.")
        return None

    # ── Estimadores ───────────────────────────────────────────
    res_naive = naive(df_clean, resultado, tratamiento)
    res_reg   = regresion(df_clean, resultado, tratamiento, covariables)
    res_gf    = g_formula(df_clean, resultado, tratamiento, covariables)
    res_ht    = ht(df_clean, resultado, tratamiento, col_ps)
    res_hajek = hajek(df_clean, resultado, tratamiento, col_ps)
    res_msm   = msm(df_clean, resultado, tratamiento, col_ps)
    res_dr    = dr(df_clean, resultado, tratamiento, covariables, col_ps)

    # ── Output ────────────────────────────────────────────────
    return {
        'n_tratados'      : n_t,
        'n_controles'     : n_c,

        'naive'           : round(res_naive, 4),

        'regresion'       : round(res_reg['ate'], 4),
        'regresion_ic95'  : (round(res_reg['ic_95'][0], 4),
                             round(res_reg['ic_95'][1], 4)),
        'regresion_pvalor': round(res_reg['p_valor'], 4),

        'g_formula'       : round(res_gf['ate'], 4),

        'ht'              : round(res_ht, 4),

        'hajek'           : round(res_hajek, 4),

        'msm'             : round(res_msm['ate'], 4),
        'msm_ic95'        : (round(res_msm['ic_95'][0], 4),
                             round(res_msm['ic_95'][1], 4)),
        'msm_pvalor'      : round(res_msm['p_valor'], 4),

        'dr'              : round(res_dr['ate'], 4),
    }
    