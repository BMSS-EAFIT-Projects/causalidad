# =============================================================================
# evaluacion.R
# Réplica del notebook evaluacion.ipynb usando el paquete causality
#
# Para correr:
#   source("evaluacion.R")
# O sección por sección en RStudio (Ctrl+Enter)
# =============================================================================

#install.packages("devtools")
options(Bioc_mirror = NULL)
options(repos = c(CRAN = "https://cloud.r-project.org"))

devtools::install_local(
  "C:/Users/afpue/OneDrive/Documentos/GitHub/causalidad/paqueteR/causality",
  dependencies = FALSE
)

library(causality)
library(MASS)   # mvrnorm — equivalente a multivariate_normal.rvs

# Función auxiliar para imprimir tabla de resultados
.imprimir_resultados <- function(res, TAU) {
  estimadores <- c("naive", "regresion", "g_formula", "ht", "hajek", "msm", "dr")
  cat(sprintf("tau verdadero = %.1f\n", TAU))
  cat(sprintf("%-14s %8s  %8s\n", "Estimador", "ATE", "Error"))
  cat(paste(rep("-", 36), collapse = ""), "\n")
  for (key in estimadores) {
    val  <- res[[key]]
    err  <- val - TAU
    flag <- if (abs(err) > 0.15) "  <- SESGO" else ""
    cat(sprintf("%-14s %8.3f  %+8.3f%s\n", key, val, err, flag))
  }
}


# =============================================================================
# ESCENARIO 1: Experimento aleatorio (RCT)
# =============================================================================

cat("\n")
cat("=============================================================\n")
cat("  ESCENARIO 1: Experimento aleatorio\n")
cat("=============================================================\n\n")

SEED <- 1234
TAU  <- 1.5

# ── Simulación de datos ──────────────────────────────────────────────────────
set.seed(SEED)

n <- 2000
p <- 0.75

# Covariables correlacionadas (slide 18)
mu    <- c(2, -3)
sigma <- matrix(c(1.23, -0.2, -0.2, 0.93), nrow = 2)
X     <- MASS::mvrnorm(n, mu = mu, Sigma = sigma)

# Tratamiento aleatorio — independiente de X
A <- rbinom(n, 1, p)

# Outcome
eps <- rnorm(n, 0, 0.2)
Y   <- 0.25 + TAU * A + 0.3 * X[, 1] + 0.15 * X[, 2] + eps

df <- data.frame(A = A, X1 = X[, 1], X2 = X[, 2], Y = Y)

tratamiento <- "A"
resultado   <- "Y"
covariables <- c("X1", "X2")

cat("Primeras filas:\n")
print(head(df, 5))

cat(sprintf("\nn = %d\n", n))
cat(sprintf("P(A=1) = %.3f  (esperado: %.2f)\n", mean(A), p))
cat(sprintf("\nMedia Y tratados:  %.4f\n", mean(Y[A == 1])))
cat(sprintf("Media Y controles: %.4f\n", mean(Y[A == 0])))
cat(sprintf("Diferencia naive:  %.4f\n", mean(Y[A == 1]) - mean(Y[A == 0])))

# ── Balance de covariables ────────────────────────────────────────────────────
cat("\nBalance de covariables (antes de balancear):\n")
cat(sprintf("  Media X1 tratados:  %.4f\n", mean(df$X1[df$A == 1])))
cat(sprintf("  Media X1 controles: %.4f\n", mean(df$X1[df$A == 0])))
cat(sprintf("  Media X2 tratados:  %.4f\n", mean(df$X2[df$A == 1])))
cat(sprintf("  Media X2 controles: %.4f\n", mean(df$X2[df$A == 0])))
cat("\nDebe estar balanceada por construccion: diferencias deben ser pequenas.\n")

# ── Propensity Score ──────────────────────────────────────────────────────────
cat("\n--- propensity_score() ---\n")
df <- propensity_score(df, tratamiento, covariables)
cat(sprintf("\nPS estimado — media: %.4f  std: %.4f\n",
            mean(df$propensity_score), sd(df$propensity_score)))
cat(sprintf("PS verdadero = %.2f para todas las unidades\n", p))


# ── ATE sin balancear ─────────────────────────────────────────────────────────
cat("\n--- ATE sin balancear ---\n")
res_orig <- calcular_ate(df, resultado, tratamiento, covariables)
.imprimir_resultados(res_orig, TAU)
cat("\nTodos deberian dar ~1.5 porque A es independiente de X.\n")


# ── Trimming ──────────────────────────────────────────────────────────────────
cat("\n--- balance(): trimming ---\n")
df_trim <- balance(df, tratamiento, metodo = "trimming", kappa = 0.05)

cat(sprintf("Filas antes:               %d\n", nrow(df)))
cat(sprintf("Filas despues del trimming: %d\n", nrow(df_trim)))
cat(sprintf("Eliminadas:                 %d\n", nrow(df) - nrow(df_trim)))
cat(sprintf("PS en df_trim — min: %.4f  max: %.4f\n",
            min(df_trim$propensity_score), max(df_trim$propensity_score)))
cat("Con un RCT el trimming no deberia eliminar a nadie.\n\n")

res_trim <- calcular_ate(df_trim, resultado, tratamiento, covariables)
.imprimir_resultados(res_trim, TAU)


# ── Truncating ────────────────────────────────────────────────────────────────
cat("\n--- balance(): truncating ---\n")
df_trunc <- balance(df, tratamiento, metodo = "truncating", kappa = 0.05)

cat(sprintf("Filas: %d  (sin eliminar)\n", nrow(df_trunc)))
cat(sprintf("PS original — min: %.4f  max: %.4f\n",
            min(df_trunc$propensity_score_original),
            max(df_trunc$propensity_score_original)))
cat(sprintf("PS truncado — min: %.4f  max: %.4f\n",
            min(df_trunc$propensity_score),
            max(df_trunc$propensity_score)))
cat("\n")

res_trunc <- calcular_ate(df_trunc, resultado, tratamiento, covariables)
.imprimir_resultados(res_trunc, TAU)


# ── Matching ──────────────────────────────────────────────────────────────────
cat("\n--- balance(): matching ---\n")
df_match <- balance(df, tratamiento, metodo = "matching")

cat(sprintf("Filas antes:               %d\n", nrow(df)))
cat(sprintf("Filas despues del matching: %d\n", nrow(df_match)))
cat(sprintf("Pares formados:             %d\n", nrow(df_match) %/% 2))
cat(sprintf("n tratados: %d\n", sum(df_match[[tratamiento]] == 1)))
cat(sprintf("n controles: %d\n", sum(df_match[[tratamiento]] == 0)))
cat("\n")

res_match <- calcular_ate(df_match, resultado, tratamiento, covariables)
.imprimir_resultados(res_match, TAU)


# ── Subclasificación ──────────────────────────────────────────────────────────
cat("\n--- balance(): subclassif ---\n")
df_sub <- balance(df, tratamiento, metodo = "subclassif", n_subclases = 5)

cat("\nDistribucion por subclase:\n")
for (sc in sort(unique(df_sub$subclase[!is.na(df_sub$subclase)]))) {
  sub <- df_sub[!is.na(df_sub$subclase) & df_sub$subclase == sc, ]
  nt  <- sum(sub[[tratamiento]] == 1)
  nc  <- sum(sub[[tratamiento]] == 0)
  cat(sprintf("  Subclase %d: n=%4d  T=%4d  C=%4d  PS_medio=%.3f\n",
              sc, nrow(sub), nt, nc, mean(sub$propensity_score)))
}
cat("\n")

res_sub <- calcular_ate(df_sub, resultado, tratamiento, covariables)
.imprimir_resultados(res_sub, TAU)


# =============================================================================
# ESCENARIO 2: Tratamiento afectado por covariables (datos observacionales)
# =============================================================================

cat("\n")
cat("=============================================================\n")
cat("  ESCENARIO 2: Tratamiento afectado por covariables\n")
cat("=============================================================\n\n")

SEED <- 1234
TAU  <- 1.0
set.seed(SEED)

n <- 2000

# Covariables correlacionadas (slide 37)
mu    <- c(1, 2)
sigma <- matrix(c(0.750, -0.375, -0.375, 1.0), nrow = 2)
X     <- MASS::mvrnorm(n, mu = mu, Sigma = sigma)

# Mecanismo de asignación — depende de X
eta  <- 0.8 * X[, 1] - 0.5 * X[, 2]
ps_v <- 1 / (1 + exp(-eta))   # PS verdadero
A    <- rbinom(n, 1, ps_v)

# Outcome — X1 y X2 son confusores
eps <- rnorm(n, 0, 0.75)
Y   <- 0.75 + TAU * A + 1.4 * X[, 1] + 5.0 * X[, 2] + eps

df <- data.frame(A = A, X1 = X[, 1], X2 = X[, 2], Y = Y, ps_verd = ps_v)

tratamiento <- "A"
resultado   <- "Y"
covariables <- c("X1", "X2")

cat("Primeras filas:\n")
print(head(df, 5))

cat(sprintf("\nn = %d\n", n))
cat(sprintf("P(A=1) = %.3f  (ya no es fijo — depende de X)\n", mean(A)))

# Desbalance de covariables
cat("\nDesbalance de covariables:\n")
cat(sprintf("  Media X1 tratados:  %.4f\n", mean(df$X1[df$A == 1])))
cat(sprintf("  Media X1 controles: %.4f  <- diferencia: %+.4f\n",
            mean(df$X1[df$A == 0]),
            mean(df$X1[df$A == 1]) - mean(df$X1[df$A == 0])))
cat(sprintf("  Media X2 tratados:  %.4f\n", mean(df$X2[df$A == 1])))
cat(sprintf("  Media X2 controles: %.4f  <- diferencia: %+.4f\n",
            mean(df$X2[df$A == 0]),
            mean(df$X2[df$A == 1]) - mean(df$X2[df$A == 0])))
cat("\nA diferencia de antes, aqui los grupos son distintos en X.\n")
cat("Esa diferencia se va a traducir en sesgo si no la controlamos.\n")

# PS verdadero
cat(sprintf("\nPS verdadero (conocido porque simulamos nosotros):\n"))
cat(sprintf("  min=%.4f  p25=%.4f  media=%.4f  p75=%.4f  max=%.4f\n",
            min(ps_v), quantile(ps_v, 0.25), mean(ps_v),
            quantile(ps_v, 0.75), max(ps_v)))
cat("\nPS verdadero por grupo:\n")
cat(sprintf("  Tratados  (A=1): media=%.4f  <- tienen PS mas alto\n",
            mean(ps_v[A == 1])))
cat(sprintf("  Controles (A=0): media=%.4f  <- tienen PS mas bajo\n",
            mean(ps_v[A == 0])))

# Diferencia naive
naive_bruto <- mean(Y[A == 1]) - mean(Y[A == 0])
cat(sprintf("\nDiferencia naive = %.4f\n", naive_bruto))
cat(sprintf("tau verdadero    = %.4f\n", TAU))
cat(sprintf("Sesgo naive      = %+.4f\n", naive_bruto - TAU))
cat("\nEl naive estima mal porque los tratados tienen X1 y X2 mas altos,\n")
cat("y X1 y X2 suben Y directamente (coefs 1.4 y 5.0).\n")


# ── Propensity Score ──────────────────────────────────────────────────────────
cat("\n--- propensity_score() ---\n")
df <- propensity_score(df, tratamiento, covariables)

cat("\nPS estimado:\n")
cat(sprintf("  min=%.4f  p25=%.4f  media=%.4f  p75=%.4f  max=%.4f\n",
            min(df$propensity_score), quantile(df$propensity_score, 0.25),
            mean(df$propensity_score), quantile(df$propensity_score, 0.75),
            max(df$propensity_score)))
cat("\nPS verdadero:\n")
cat(sprintf("  min=%.4f  p25=%.4f  media=%.4f  p75=%.4f  max=%.4f\n",
            min(ps_v), quantile(ps_v, 0.25), mean(ps_v),
            quantile(ps_v, 0.75), max(ps_v)))

corr <- cor(df$propensity_score, df$ps_verd)
cat(sprintf("\nCorrelacion PS estimado vs PS verdadero: %.4f\n", corr))
cat("Una correlacion alta (>0.99) indica que el modelo logistico\n")
cat("capturo bien el mecanismo de asignacion.\n")


# ── ATE sin balancear ─────────────────────────────────────────────────────────
cat("\n--- ATE sin balancear ---\n")
res_orig <- calcular_ate(df, resultado, tratamiento, covariables)
.imprimir_resultados(res_orig, TAU)
cat("\nEl naive deberia estar sesgado.\n")
cat("Regresion, g-formula y DR deben estar cerca de 1.0 (controlan X1 y X2).\n")
cat("HT y Hajek dependen del PS estimado.\n")


# ── Trimming ──────────────────────────────────────────────────────────────────
cat("\n--- balance(): trimming ---\n")
df_trim <- balance(df, tratamiento, metodo = "trimming", kappa = 0.05)

cat(sprintf("Filas antes:  %d\n", nrow(df)))
cat(sprintf("Filas despues: %d\n", nrow(df_trim)))
cat(sprintf("Eliminadas:   %d\n", nrow(df) - nrow(df_trim)))
cat("\nA diferencia del RCT, aqui si hay PS extremos porque A depende de X.\n")
cat("El trimming elimina unidades sin contrafactual comparable.\n")

cat("\nDesbalance DESPUES del trimming:\n")
cat(sprintf("  Media X1 tratados:  %.4f\n", mean(df_trim$X1[df_trim$A == 1])))
cat(sprintf("  Media X1 controles: %.4f  <- diferencia: %+.4f\n",
            mean(df_trim$X1[df_trim$A == 0]),
            mean(df_trim$X1[df_trim$A == 1]) - mean(df_trim$X1[df_trim$A == 0])))
cat(sprintf("  Media X2 tratados:  %.4f\n", mean(df_trim$X2[df_trim$A == 1])))
cat(sprintf("  Media X2 controles: %.4f  <- diferencia: %+.4f\n",
            mean(df_trim$X2[df_trim$A == 0]),
            mean(df_trim$X2[df_trim$A == 1]) - mean(df_trim$X2[df_trim$A == 0])))
cat("\n")

res_trim <- calcular_ate(df_trim, resultado, tratamiento, covariables)
.imprimir_resultados(res_trim, TAU)


# ── Truncating ────────────────────────────────────────────────────────────────
cat("\n--- balance(): truncating ---\n")
df_trunc <- balance(df, tratamiento, metodo = "truncating", kappa = 0.05)

n_truncados <- sum(df$propensity_score < 0.05 | df$propensity_score > 0.95)
cat(sprintf("Filas: %d  (sin eliminar)\n", nrow(df_trunc)))
cat(sprintf("PS truncados: %d\n", n_truncados))
cat(sprintf("PS original — min: %.4f  max: %.4f\n",
            min(df_trunc$propensity_score_original),
            max(df_trunc$propensity_score_original)))
cat(sprintf("PS truncado — min: %.4f  max: %.4f\n",
            min(df_trunc$propensity_score),
            max(df_trunc$propensity_score)))
cat("\n")

res_trunc <- calcular_ate(df_trunc, resultado, tratamiento, covariables)
.imprimir_resultados(res_trunc, TAU)


# ── Matching ──────────────────────────────────────────────────────────────────
cat("\n--- balance(): matching ---\n")
df_match <- balance(df, tratamiento, metodo = "matching")

cat(sprintf("Filas antes:  %d\n", nrow(df)))
cat(sprintf("Filas despues: %d\n", nrow(df_match)))
cat(sprintf("Pares formados: %d\n", nrow(df_match) %/% 2))

cat("\nDesbalance DESPUES del matching:\n")
cat(sprintf("  Media X1 tratados:  %.4f\n", mean(df_match$X1[df_match$A == 1])))
cat(sprintf("  Media X1 controles: %.4f  <- diferencia: %+.4f\n",
            mean(df_match$X1[df_match$A == 0]),
            mean(df_match$X1[df_match$A == 1]) - mean(df_match$X1[df_match$A == 0])))
cat(sprintf("  Media X2 tratados:  %.4f\n", mean(df_match$X2[df_match$A == 1])))
cat(sprintf("  Media X2 controles: %.4f  <- diferencia: %+.4f\n",
            mean(df_match$X2[df_match$A == 0]),
            mean(df_match$X2[df_match$A == 1]) - mean(df_match$X2[df_match$A == 0])))
cat("El matching deberia reducir el desbalance acercando controles a tratados en PS.\n\n")

res_match <- calcular_ate(df_match, resultado, tratamiento, covariables)
.imprimir_resultados(res_match, TAU)


# ── Subclasificación ──────────────────────────────────────────────────────────
cat("\n--- balance(): subclassif ---\n")
df_sub <- balance(df, tratamiento, metodo = "subclassif", n_subclases = 5)

cat("\nDistribucion por subclase:\n")
for (sc in sort(unique(df_sub$subclase[!is.na(df_sub$subclase)]))) {
  sub <- df_sub[!is.na(df_sub$subclase) & df_sub$subclase == sc, ]
  nt  <- sum(sub[[tratamiento]] == 1)
  nc  <- sum(sub[[tratamiento]] == 0)
  cat(sprintf("  Subclase %d: n=%4d  T=%4d  C=%4d  PS_medio=%.3f\n",
              sc, nrow(sub), nt, nc, mean(sub$propensity_score)))
}
cat("\nA diferencia del RCT, aqui la subclase 1 (PS bajo) tiene pocos tratados\n")
cat("y la subclase 5 (PS alto) tiene pocos controles.\n\n")

res_sub <- calcular_ate(df_sub, resultado, tratamiento, covariables)
.imprimir_resultados(res_sub, TAU)

cat("\n=============================================================\n")
cat("  FIN\n")
cat("=============================================================\n")
