# %% [markdown]
# # Simulaciones de inferencia causal
# Escenarios extraídos directamente de las diapositivas de LaLonde (Taddeo, 2025).
#
# Escenarios
# ----------
# 1. RCT                  — slides 18-21  — A independiente de X, τ=1.5
# 2. Observacional        — slide 37      — A depende de X, τ=1.0, datos completos
# 3. Variable omitida     — slide 38      — misma DGP pero X2 removida del modelo
# 4. Modelo mal especificado — slide 42   — outcome con |X|, modelo ajusta lineal
# 5. Heterogeneidad       — slide 40      — efecto varía por subgrupo S

# %%
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import causalidad as cs

SEED = 1234   # mismo seed que los slides
N    = 2000   # n=2000 como en el slide 18

rng = np.random.default_rng(SEED)

def imprimir_resultados(nombre, res, tau_verdadero):
    if res is None:
        print(f"  {nombre:<22} — sin resultados")
        return
    print(f"\n  {'─'*55}")
    print(f"  Balanceo: {nombre}")
    print(f"  n tratados={res['n_tratados']}  n controles={res['n_controles']}")
    print(f"  {'Estimador':<14} {'ATE':>8}  {'Error':>8}  {'Sesgo?':>8}")
    print(f"  {'─'*55}")
    for key in ['naive', 'regresion', 'g_formula', 'ht', 'hajek', 'msm', 'dr']:
        val  = res[key]
        err  = val - tau_verdadero
        flag = '  ← SESGO' if abs(err) > 0.15 else ''
        print(f"  {key:<14} {val:>8.3f}  {err:>+8.3f}{flag}")

def pipeline_completo(df, tratamiento, resultado, covariables, tau_verdadero,
                      titulo):
    """Estima PS, balancea por 4 métodos y calcula ATE con cada uno."""
    print(f"\n{'═'*60}")
    print(f"  {titulo}")
    print(f"  τ verdadero = {tau_verdadero}")
    print(f"{'═'*60}")

    df = cs.propensity_score(df, tratamiento, covariables)

    df_trim  = cs.balance(df, tratamiento, metodo='trimming')
    df_trunc = cs.balance(df, tratamiento, metodo='truncating')
    df_match = cs.balance(df, tratamiento, metodo='matching')
    df_sub   = cs.balance(df, tratamiento, metodo='subclassif')

    for nombre, datos in [('Sin balanceo',      df),
                           ('Trimming',         df_trim),
                           ('Truncating',       df_trunc),
                           ('Matching',         df_match),
                           ('Subclasificación', df_sub)]:
        res = cs.calcular_ate(datos, resultado, tratamiento, covariables)
        imprimir_resultados(nombre, res, tau_verdadero)


# %% [markdown]
# ---
# ## ESCENARIO 1 — Experimento aleatorio (RCT)
# Slide 18/88
#
# Especificación exacta del slide:
#   n = 2000
#   p = 0.75  (probabilidad de tratamiento, alta e imbalanceada a propósito)
#   (X1, X2) ~ N(μ=(2,-3), Σ=[[1.23,-0.2],[-0.2,0.93]])  — correlacionadas
#   A ~ Bernoulli(0.75)  — INDEPENDIENTE de X
#   Y = 0.25 + 1.5·A + 0.3·X1 + 0.15·X2 + ε,  ε ~ N(0, 0.04)
#   τ = 1.5
#
# Resultado esperado (slide 21):
#   Naive     ≈ 1.49
#   Regresión ≈ 1.48
#   Todos los estimadores deberían recuperar τ=1.5 porque A ⊥⊥ X

# %%
np.random.seed(SEED)

tau_1 = 1.5
n_1   = 2000
p_1   = 0.75

# Covariables correlacionadas — exactamente como en el slide
mu    = [2, -3]
sigma = [[1.23, -0.2], [-0.2, 0.93]]
X_1   = multivariate_normal.rvs(mean=mu, cov=sigma, size=n_1,
                                 random_state=SEED)

# A independiente de X
A_1 = np.random.binomial(1, p_1, n_1)

# Outcome
eps_1 = np.random.normal(0, 0.2, n_1)   # σ=0.2 → varianza=0.04
Y_1   = 0.25 + tau_1 * A_1 + 0.3 * X_1[:,0] + 0.15 * X_1[:,1] + eps_1

df_1 = pd.DataFrame({'A': A_1, 'X1': X_1[:,0], 'X2': X_1[:,1], 'Y': Y_1})

print(f"P(A=1)={A_1.mean():.3f}  (esperado 0.75)")
print(f"Naive = {Y_1[A_1==1].mean() - Y_1[A_1==0].mean():.3f}  (slide: 1.49)")

pipeline_completo(df_1, 'A', 'Y', ['X1', 'X2'], tau_1,
                  "ESCENARIO 1 — RCT (A ⊥⊥ X)")


# %% [markdown]
# ---
# ## ESCENARIO 2 — Estudio observacional, datos completos
# Slide 37/88
#
# Especificación exacta del slide:
#   τ = 1
#   (X1, X2) ~ N(μ=(1,2), Σ=[[0.75,-0.375],[-0.375,1]])
#   η(x) = 0.8·X1 - 0.5·X2
#   P(A=1|X) = sigmoid(η(X))   — A depende de X (confusión!)
#   Y = 0.75 + τ·A + 1.4·X1 + 5·X2 + ε,  ε ~ N(0, 0.75²)
#
# Con datos completos (X1 y X2) la regresión recupera τ=1.
# Resultado del slide (n=40): τ̂ ≈ 1.07
#
# El naive estará sesgado porque A y X están correlacionados.

# %%
np.random.seed(SEED)

tau_2 = 1.0

mu_2    = [1, 2]
sigma_2 = [[0.750, -0.375], [-0.375, 1.0]]
X_2     = multivariate_normal.rvs(mean=mu_2, cov=sigma_2, size=N,
                                   random_state=SEED)

# Tratamiento depende de X — mecanismo logístico del slide
eta_2 = 0.8 * X_2[:,0] - 0.5 * X_2[:,1]
ps_2  = 1 / (1 + np.exp(-eta_2))
A_2   = np.random.binomial(1, ps_2)

# Outcome — X1 y X2 son confusores
eps_2 = np.random.normal(0, 0.75, N)
Y_2   = 0.75 + tau_2 * A_2 + 1.4 * X_2[:,0] + 5.0 * X_2[:,1] + eps_2

df_2 = pd.DataFrame({'A': A_2, 'X1': X_2[:,0], 'X2': X_2[:,1], 'Y': Y_2})

print(f"P(A=1) = {A_2.mean():.3f}")
print(f"Naive  = {Y_2[A_2==1].mean() - Y_2[A_2==0].mean():.3f}  (esperado: sesgado)")

pipeline_completo(df_2, 'A', 'Y', ['X1', 'X2'], tau_2,
                  "ESCENARIO 2 — Observacional, datos completos")


# %% [markdown]
# ---
# ## ESCENARIO 3 — Variable omitida (endogeneidad)
# Slide 38/88
#
# MISMO DGP que el escenario 2, pero X2 se omite del modelo.
# Esto replica el "Fit 2 (Partial Data: X2 removed)" del slide.
#
# Resultado del slide (n=40): τ̂ ≈ 1.56 (50% sobreestimado)
#
# X2 es confusora (afecta tanto A como Y) y al omitirla el estimador
# absorbe su efecto en el coeficiente de A → sesgo por variable omitida.

# %%
# Mismo df_2, pero ahora covariables SOLO con X1 (omitimos X2)
print("Nota: mismos datos que Escenario 2 pero X2 omitida del modelo.")
print(f"      El ATE verdadero sigue siendo τ = {tau_2}")

pipeline_completo(df_2, 'A', 'Y', ['X1'],   # X2 omitida
                  tau_2,
                  "ESCENARIO 3 — Variable omitida (X2 removida)")


# %% [markdown]
# ---
# ## ESCENARIO 4 — Modelo mal especificado
# Slide 42/88
#
# Especificación exacta del slide:
#   Y = 0.75 + A + 4·|X| + 0.5·ε      ← outcome NO LINEAL en X
#   η(x) = -0.5 + 2X
#   P(A=1|X) = sigmoid(η(X))
#   τ = 1  (ATE verdadero)
#
# Modelo correcto:  Y ~ A + |X|   → recupera τ=1
# Modelo mal especificado: Y ~ A + X  (lineal en X, no en |X|)
#                          → sesgo aunque X sea observada
#
# Lección del slide: la especificación incorrecta del modelo de outcome
# genera sesgo AUN CUANDO todos los confusores están disponibles.

# %%
np.random.seed(SEED)

tau_4 = 1.0

X_4  = np.random.normal(0, 1, N)

# Tratamiento: η(x) = -0.5 + 2X
eta_4 = -0.5 + 2 * X_4
ps_4  = 1 / (1 + np.exp(-eta_4))
A_4   = np.random.binomial(1, ps_4)

# Outcome NO LINEAL: 4·|X|
eps_4 = np.random.normal(0, 0.5, N)
Y_4   = 0.75 + tau_4 * A_4 + 4 * np.abs(X_4) + eps_4

# Creamos dos DataFrames:
# df_4_correcto : incluye |X| como covariable (modelo correcto)
# df_4_mal      : solo tiene X lineal (modelo mal especificado)
df_4 = pd.DataFrame({'A': A_4, 'X': X_4, 'absX': np.abs(X_4), 'Y': Y_4})

print(f"Naive = {Y_4[A_4==1].mean() - Y_4[A_4==0].mean():.3f}  (esperado: muy sesgado)")

print("\n── Modelo CORRECTO (incluye |X|) ────────────────────")
pipeline_completo(df_4, 'A', 'Y', ['X', 'absX'], tau_4,
                  "ESCENARIO 4a — Modelo correcto (incluye |X|)")

print("\n── Modelo MAL ESPECIFICADO (solo X lineal) ──────────")
pipeline_completo(df_4, 'A', 'Y', ['X'], tau_4,
                  "ESCENARIO 4b — Modelo mal especificado (solo X lineal)")


# %% [markdown]
# ---
# ## ESCENARIO 5 — Heterogeneidad del efecto
# Slides 40-41/88
#
# Especificación:
#   X = (S, W) donde S es binaria (modificador de efecto), W continua
#   Y = τ·A + δ·(A·S) + α·W + ε
#   τ = 1.0  (efecto para S=0)
#   δ = 1.5  (efecto adicional para S=1)
#
#   ATE(S=0) = τ     = 1.0
#   ATE(S=1) = τ + δ = 2.5
#   ATE marginal     = τ + δ·P(S=1)
#
# Los estimadores sin interacción recuperan el ATE marginal (correcto).
# El naive estará sesgado si S también afecta el tratamiento.

# %%
np.random.seed(SEED)

tau_5 = 1.0   # efecto base
delta = 1.5   # efecto adicional para S=1

S_5 = np.random.binomial(1, 0.5, N)   # modificador de efecto, P(S=1)=0.5
W_5 = np.random.normal(0, 1, N)       # covariable continua

# ATE verdadero marginal: τ + δ·P(S=1) = 1.0 + 1.5·0.5 = 1.75
tau_marginal = tau_5 + delta * S_5.mean()

# Tratamiento depende de S (sesgo de selección por S)
eta_5 = -0.5 + 1.5 * S_5 - 0.3 * W_5
ps_5  = 1 / (1 + np.exp(-eta_5))
A_5   = np.random.binomial(1, ps_5)

# Outcome con efecto heterogéneo
eps_5 = np.random.normal(0, 1, N)
Y_5   = tau_5 * A_5 + delta * (A_5 * S_5) + 1.2 * W_5 + eps_5

df_5 = pd.DataFrame({'A': A_5, 'S': S_5, 'W': W_5, 'Y': Y_5})

print(f"P(A=1|S=0) = {A_5[S_5==0].mean():.3f}")
print(f"P(A=1|S=1) = {A_5[S_5==1].mean():.3f}")
print(f"ATE(S=0)   = {tau_5:.2f}")
print(f"ATE(S=1)   = {tau_5 + delta:.2f}")
print(f"ATE marginal verdadero = {tau_marginal:.3f}")

print("\n── Sin interacción (recupera ATE marginal) ──────────")
pipeline_completo(df_5, 'A', 'Y', ['S', 'W'], tau_marginal,
                  "ESCENARIO 5a — Heterogeneidad, sin interacción")

print("\n── Con interacción A*S (recupera ATE por subgrupo) ──")
df_5['A_x_S'] = df_5['A'] * df_5['S']
pipeline_completo(df_5, 'A', 'Y', ['S', 'W', 'A_x_S'], tau_marginal,
                  "ESCENARIO 5b — Heterogeneidad, con interacción A*S")