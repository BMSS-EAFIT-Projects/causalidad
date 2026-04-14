#' Estimar el Propensity Score
#'
#' Estima pi(X) = P(A=1|X) con regresión logística y agrega el resultado
#' como columna al data frame. Equivalente al `glm(..., family="binomial")`
#' de los slides de LaLonde (Taddeo, 2025, p.68 y p.112).
#'
#' @param df Data frame con los datos.
#' @param tratamiento Nombre de la columna de tratamiento (valores 0/1).
#' @param covariables Vector de caracteres con los nombres de las covariables.
#' @param columna Nombre de la columna donde guardar el PS
#'   (default `"propensity_score"`).
#'
#' @return Data frame idéntico al original con la columna del PS añadida.
#'   El PS está garantizado en el rango (1e-8, 1-1e-8) para evitar
#'   divisiones por cero en los estimadores IPW.
#'
#' @details
#' Equivalente en R de los slides (p.68):
#' ```r
#' PS_fit <- glm(treatment ~ ., data = df, family = "binomial")
#' df$propensity_score <- predict(PS_fit, type = "response")
#' ```
#'
#' Si incluyes términos polinómicos (como en los slides p.112), pásalos
#' directamente en `covariables`:
#' ```r
#' covs <- c("age", "education", "I(education^2)", "black", "hispanic")
#' df <- propensity_score(df, "treatment", covs)
#' ```
#'
#' @examples
#' data(lalonde, package = "MatchIt")
#' covs <- c("age", "educ", "black", "hispan", "married", "nodegree",
#'           "re74", "re75")
#' df <- propensity_score(lalonde, tratamiento = "treat", covariables = covs)
#' summary(df$propensity_score)
#'
#' @export
propensity_score <- function(df, tratamiento, covariables,
                              columna = "propensity_score") {

  # ── Validaciones ──────────────────────────────────────────────────────────
  cols_req <- c(tratamiento, covariables)
  # Para columnas con expresiones tipo "I(x^2)" extraemos solo la parte base
  cols_simples <- covariables[!grepl("[^a-zA-Z0-9_\\.]", covariables)]
  cols_faltantes <- setdiff(c(tratamiento, cols_simples), names(df))

  if (length(cols_faltantes) > 0) {
    stop(paste("Columnas no encontradas en df:",
               paste(cols_faltantes, collapse = ", ")))
  }

  if (length(unique(df[[tratamiento]])) < 2) {
    stop("El tratamiento no tiene variacion: todos los valores son iguales.")
  }

  if (columna %in% names(df)) {
    message(sprintf("Aviso: la columna '%s' ya existe y sera sobreescrita.",
                    columna))
  }

  # ── Estimación ────────────────────────────────────────────────────────────
  formula_str <- paste(tratamiento, "~", paste(covariables, collapse = " + "))
  formula_obj <- stats::as.formula(formula_str)

  df_clean <- df[stats::complete.cases(df[, c(tratamiento, cols_simples)]), ]

  modelo <- stats::glm(formula_obj, data = df_clean, family = stats::binomial())

  ps_vals <- stats::predict(modelo, newdata = df_clean, type = "response")
  ps_vals <- pmin(pmax(ps_vals, 1e-8), 1 - 1e-8)

  # Guardar en el df original (NA para filas con datos faltantes)
  df[[columna]] <- NA_real_
  df[rownames(df_clean), columna] <- ps_vals

  # ── Diagnóstico ───────────────────────────────────────────────────────────
  n_t <- sum(df_clean[[tratamiento]] == 1, na.rm = TRUE)
  n_c <- sum(df_clean[[tratamiento]] == 0, na.rm = TRUE)

  message(sprintf(
    "propensity_score() -- n=%d (T=%d, C=%d)",
    nrow(df_clean), n_t, n_c
  ))
  message(sprintf(
    "  PS: min=%.4f  p25=%.4f  media=%.4f  p75=%.4f  max=%.4f",
    min(ps_vals), stats::quantile(ps_vals, 0.25),
    mean(ps_vals), stats::quantile(ps_vals, 0.75),
    max(ps_vals)
  ))

  n_extremos <- sum(ps_vals < 0.05 | ps_vals > 0.95)
  if (n_extremos > 0) {
    pct <- round(n_extremos / length(ps_vals) * 100, 1)
    message(sprintf(
      "  Aviso: %d unidades (%.1f%%) con PS < 0.05 o PS > 0.95 -- considera trimming o truncating.",
      n_extremos, pct
    ))
  }

  return(df)
}
