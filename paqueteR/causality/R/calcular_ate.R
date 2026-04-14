#' Calcular el ATE con Múltiples Estimadores
#'
#' Calcula el Average Treatment Effect (ATE) usando todos los estimadores
#' disponibles. El data frame debe tener la columna del PS ya calculada
#' (output de [propensity_score()] y opcionalmente [balance()]).
#'
#' Estimadores implementados (fiel a los slides de LaLonde / Taddeo, 2025):
#' 1. **naive**: Diferencia de medias E[Y|A=1] - E[Y|A=0] (slide 5/9)
#' 2. **regresion**: OLS: Y ~ A + X, coeficiente de A (slide 5/9)
#' 3. **g_formula**: Estandarización: promedio de Ŷ(1) - Ŷ(0) (slide 87)
#' 4. **ht**: Horvitz-Thompson (IPW no normalizado) (slide 54)
#' 5. **hajek**: Hajek (IPW normalizado) (slide 65)
#' 6. **msm**: Modelo Estructural Marginal (WLS pesos estabilizados) (slide 77)
#' 7. **dr**: Estimador Doblemente Robusto / AIPW (slide 78)
#'
#' @param df Data frame balanceado con columna del PS ya calculada.
#' @param resultado Nombre de la columna de resultado.
#' @param tratamiento Nombre de la columna de tratamiento (valores 0/1).
#' @param covariables Vector de nombres de covariables (para regresión,
#'   g-fórmula y DR).
#' @param col_ps Nombre de la columna del PS (default `"propensity_score"`).
#'
#' @return Lista con los ATE de todos los estimadores:
#'   - `n_tratados`, `n_controles`
#'   - `naive`
#'   - `regresion`, `regresion_ic95`, `regresion_pvalor`
#'   - `g_formula`
#'   - `ht`
#'   - `hajek`
#'   - `msm`, `msm_ic95`, `msm_pvalor`
#'   - `dr`
#'
#' @examples
#' data(lalonde, package = "MatchIt")
#' covs <- c("age", "educ", "black", "hispan", "married", "nodegree",
#'           "re74", "re75")
#' df      <- propensity_score(lalonde, "treat", covs)
#' df_trim <- balance(df, "treat", metodo = "trimming")
#' res     <- calcular_ate(df_trim, resultado = "re78",
#'                         tratamiento = "treat", covariables = covs)
#' cat("DR:   ", res$dr, "\n")
#' cat("Hajek:", res$hajek, "\n")
#'
#' @export
calcular_ate <- function(df, resultado, tratamiento, covariables,
                          col_ps = "propensity_score") {

  # ── Validaciones ──────────────────────────────────────────────────────────
  if (!col_ps %in% names(df)) {
    stop(sprintf(
      "La columna '%s' no existe. Llama primero a propensity_score().", col_ps
    ))
  }

  cols_req  <- c(resultado, tratamiento, covariables)
  faltantes <- setdiff(cols_req, names(df))
  if (length(faltantes) > 0) {
    stop(paste("Columnas no encontradas en df:", paste(faltantes, collapse = ", ")))
  }

  df_clean <- df[stats::complete.cases(df[, c(cols_req, col_ps)]), ]

  if (nrow(df_clean) == 0) {
    message("calcular_ate() -- Error: sin datos completos.")
    return(NULL)
  }

  n_t <- sum(df_clean[[tratamiento]] == 1)
  n_c <- sum(df_clean[[tratamiento]] == 0)

  if (n_t == 0 || n_c == 0) {
    message("calcular_ate() -- Error: sin variacion en el tratamiento.")
    return(NULL)
  }

  # ── Estimadores ───────────────────────────────────────────────────────────
  res_naive <- .ate_naive(df_clean, resultado, tratamiento)
  res_reg   <- .ate_regresion(df_clean, resultado, tratamiento, covariables)
  res_gf    <- .ate_g_formula(df_clean, resultado, tratamiento, covariables)
  res_ht    <- .ate_ht(df_clean, resultado, tratamiento, col_ps)
  res_hajek <- .ate_hajek(df_clean, resultado, tratamiento, col_ps)
  res_msm   <- .ate_msm(df_clean, resultado, tratamiento, col_ps)
  res_dr    <- .ate_dr(df_clean, resultado, tratamiento, covariables, col_ps)

  # ── Output ────────────────────────────────────────────────────────────────
  list(
    n_tratados        = n_t,
    n_controles       = n_c,

    naive             = round(res_naive, 4),

    regresion         = round(res_reg$ate, 4),
    regresion_ic95    = round(res_reg$ic95, 4),
    regresion_pvalor  = round(res_reg$pvalor, 4),

    g_formula         = round(res_gf$ate, 4),

    ht                = round(res_ht, 4),

    hajek             = round(res_hajek, 4),

    msm               = round(res_msm$ate, 4),
    msm_ic95          = round(res_msm$ic95, 4),
    msm_pvalor        = round(res_msm$pvalor, 4),

    dr                = round(res_dr, 4)
  )
}


# ══════════════════════════════════════════════════════════════════════════════
# ESTIMADORES INTERNOS
# ══════════════════════════════════════════════════════════════════════════════

# 1. NAIVE ─────────────────────────────────────────────────────────────────
#' @keywords internal
.ate_naive <- function(df, resultado, tratamiento) {
  y1 <- df[[resultado]][df[[tratamiento]] == 1]
  y0 <- df[[resultado]][df[[tratamiento]] == 0]
  mean(y1) - mean(y0)
}


# 2. REGRESIÓN ─────────────────────────────────────────────────────────────
#' @keywords internal
.ate_regresion <- function(df, resultado, tratamiento, covariables) {
  formula_str <- paste(resultado, "~", tratamiento, "+",
                       paste(covariables, collapse = " + "))
  modelo <- stats::lm(stats::as.formula(formula_str), data = df)
  ci     <- stats::confint(modelo)[tratamiento, ]
  list(
    ate    = stats::coef(modelo)[tratamiento],
    ic95   = as.numeric(ci),
    pvalor = summary(modelo)$coefficients[tratamiento, "Pr(>|t|)"]
  )
}


# 3. G-FÓRMULA ─────────────────────────────────────────────────────────────
#' @keywords internal
.ate_g_formula <- function(df, resultado, tratamiento, covariables) {
  formula_str <- paste(resultado, "~", tratamiento, "+",
                       paste(covariables, collapse = " + "))

  valores_unicos <- unique(stats::na.omit(df[[resultado]]))
  es_binario     <- all(valores_unicos %in% c(0, 1))

  if (es_binario) {
    modelo      <- stats::glm(stats::as.formula(formula_str),
                              data = df, family = stats::binomial())
    modelo_used <- "logit"
  } else {
    modelo      <- stats::lm(stats::as.formula(formula_str), data = df)
    modelo_used <- "ols"
  }

  df1 <- df; df1[[tratamiento]] <- 1
  df0 <- df; df0[[tratamiento]] <- 0

  y1_hat <- stats::predict(modelo, newdata = df1, type = "response")
  y0_hat <- stats::predict(modelo, newdata = df0, type = "response")
  efecto <- y1_hat - y0_hat

  list(
    ate          = mean(efecto),
    modelo_usado = modelo_used,
    efectos_ind  = efecto
  )
}


# 4. HORVITZ-THOMPSON ─────────────────────────────────────────────────────
#' @keywords internal
.ate_ht <- function(df, resultado, tratamiento, col_ps) {
  A  <- as.numeric(df[[tratamiento]])
  Y  <- as.numeric(df[[resultado]])
  ps <- as.numeric(df[[col_ps]])

  mean(A * Y / ps) - mean((1 - A) * Y / (1 - ps))
}


# 5. HAJEK ────────────────────────────────────────────────────────────────
#' @keywords internal
.ate_hajek <- function(df, resultado, tratamiento, col_ps) {
  A  <- as.numeric(df[[tratamiento]])
  Y  <- as.numeric(df[[resultado]])
  ps <- as.numeric(df[[col_ps]])

  mu1 <- sum(A * Y / ps)       / sum(A / ps)
  mu0 <- sum((1 - A) * Y / (1 - ps)) / sum((1 - A) / (1 - ps))
  mu1 - mu0
}


# 6. MSM ──────────────────────────────────────────────────────────────────
#' @keywords internal
.ate_msm <- function(df, resultado, tratamiento, col_ps) {
  A  <- as.numeric(df[[tratamiento]])
  ps <- as.numeric(df[[col_ps]])

  # Pesos estabilizados: W̃ᵢ = E(A) / π(Xᵢ)  [slide 77]
  e_a   <- mean(A)
  pesos <- ifelse(A == 1, e_a / ps, (1 - e_a) / (1 - ps))

  formula_str <- paste(resultado, "~", tratamiento)
  modelo      <- stats::lm(stats::as.formula(formula_str),
                            data = df, weights = pesos)
  ci          <- stats::confint(modelo)[tratamiento, ]

  list(
    ate    = stats::coef(modelo)[tratamiento],
    ic95   = as.numeric(ci),
    pvalor = summary(modelo)$coefficients[tratamiento, "Pr(>|t|)"]
  )
}


# 7. DOBLEMENTE ROBUSTO (AIPW) ────────────────────────────────────────────
#' @keywords internal
.ate_dr <- function(df, resultado, tratamiento, covariables, col_ps) {
  formula_str <- paste(resultado, "~", tratamiento, "+",
                       paste(covariables, collapse = " + "))
  modelo <- stats::lm(stats::as.formula(formula_str), data = df)

  df1 <- df; df1[[tratamiento]] <- 1
  df0 <- df; df0[[tratamiento]] <- 0

  mu1 <- stats::predict(modelo, newdata = df1)
  mu0 <- stats::predict(modelo, newdata = df0)

  A  <- as.numeric(df[[tratamiento]])
  Y  <- as.numeric(df[[resultado]])
  ps <- as.numeric(df[[col_ps]])

  # Fórmula AIPW (slide 78):
  # m1ᵢ = A(Y - μ₁) / π  +  μ₁
  # m0ᵢ = (1-A)(Y - μ₀) / (1-π)  +  μ₀
  m1 <- A * (Y - mu1) / ps + mu1
  m0 <- (1 - A) * (Y - mu0) / (1 - ps) + mu0

  mean(m1) - mean(m0)
}
