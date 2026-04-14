#' Balanceo por Propensity Score
#'
#' Balancea una muestra observacional usando el propensity score almacenado
#' en una columna del data frame. Requiere haber llamado antes a
#' [propensity_score()].
#'
#' Métodos disponibles (fiel a los slides de LaLonde / Taddeo, 2025):
#' - `"matching"`: matching 1:1 vecino más cercano en PS
#' - `"subclassif"`: subclasificación por quintiles de PS
#' - `"trimming"`: eliminar unidades con PS < kappa o PS > 1-kappa (slide 65)
#' - `"truncating"`: recortar PS a max{kappa, min{PS, 1-kappa}} (slide 65)
#'
#' @param df Data frame con la columna del PS ya calculada.
#' @param tratamiento Nombre de la columna de tratamiento (valores 0/1).
#' @param metodo Método de balanceo: `"matching"`, `"subclassif"`,
#'   `"trimming"` (default), o `"truncating"`.
#' @param col_ps Nombre de la columna del PS (default `"propensity_score"`).
#' @param kappa Umbral para trimming y truncating (default `0.05`).
#'   Rango válido: (0, 0.5). Los slides recomiendan 0.05-0.10.
#' @param n_subclases Número de subclases para `"subclassif"` (default `5`).
#' @param replacement Para `"matching"`: permitir reemplazo (default `FALSE`).
#' @param caliper Para `"matching"`: distancia máxima aceptable en PS
#'   (default `NULL` = sin restricción).
#'
#' @return Data frame balanceado. Columnas adicionales según método:
#'   - `"matching"`: `pair_id` (ID del par tratado-control)
#'   - `"subclassif"`: `subclase` (1..n_subclases), `peso_subclase`
#'   - `"truncating"`: `{col_ps}_original` (PS antes de truncar)
#'   - `"trimming"`: sin columnas extra (solo menos filas)
#'
#' @examples
#' data(lalonde, package = "MatchIt")
#' covs <- c("age", "educ", "black", "hispan", "married", "nodegree",
#'           "re74", "re75")
#' df <- propensity_score(lalonde, "treat", covs)
#'
#' df_trim    <- balance(df, "treat", metodo = "trimming")
#' df_trunc   <- balance(df, "treat", metodo = "truncating")
#' df_match   <- balance(df, "treat", metodo = "matching")
#' df_subclf  <- balance(df, "treat", metodo = "subclassif")
#'
#' @export
balance <- function(df, tratamiento,
                    metodo    = "trimming",
                    col_ps    = "propensity_score",
                    kappa     = 0.05,
                    n_subclases = 5,
                    replacement = FALSE,
                    caliper   = NULL) {

  # ── Validaciones ──────────────────────────────────────────────────────────
  if (!col_ps %in% names(df)) {
    stop(sprintf(
      "La columna '%s' no existe en df. Llama primero a propensity_score().",
      col_ps
    ))
  }

  metodos_validos <- c("matching", "subclassif", "trimming", "truncating")
  if (!metodo %in% metodos_validos) {
    stop(sprintf("metodo='%s' no reconocido. Opciones: %s",
                 metodo, paste(metodos_validos, collapse = ", ")))
  }

  if (metodo %in% c("trimming", "truncating") && !(kappa > 0 && kappa < 0.5)) {
    stop(sprintf("kappa debe estar en (0, 0.5). Recibido: %g", kappa))
  }

  if (length(unique(stats::na.omit(df[[tratamiento]]))) < 2) {
    message("balance() -- Error: sin variacion en el tratamiento.")
    return(NULL)
  }

  # Descartar filas sin PS
  df_clean <- df[!is.na(df[[col_ps]]), ]
  n_drop   <- nrow(df) - nrow(df_clean)
  if (n_drop > 0) {
    message(sprintf("balance() -- %d filas sin PS descartadas.", n_drop))
  }

  # ── Encabezado ────────────────────────────────────────────────────────────
  n_t <- sum(df_clean[[tratamiento]] == 1, na.rm = TRUE)
  n_c <- sum(df_clean[[tratamiento]] == 0, na.rm = TRUE)
  sep <- paste(rep("-", 50), collapse = "")
  message(sep)
  message(sprintf("  balance() | metodo='%s' | n=%d (T=%d, C=%d)",
                  metodo, nrow(df_clean), n_t, n_c))
  message(sep)

  # ── Despachar al método ───────────────────────────────────────────────────
  resultado <- switch(metodo,
    matching   = .balance_matching(df_clean, tratamiento, col_ps,
                                   replacement, caliper),
    subclassif = .balance_subclassification(df_clean, tratamiento, col_ps,
                                            n_subclases),
    trimming   = .balance_trimming(df_clean, tratamiento, col_ps, kappa),
    truncating = .balance_truncating(df_clean, tratamiento, col_ps, kappa)
  )

  message(sep)
  message("")
  return(resultado)
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS INTERNOS (no exportados)
# ══════════════════════════════════════════════════════════════════════════════

#' @keywords internal
.balance_matching <- function(df, tratamiento, col_ps, replacement, caliper) {

  idx_t <- which(df[[tratamiento]] == 1)
  idx_c <- which(df[[tratamiento]] == 0)
  ps    <- df[[col_ps]]

  matched_t <- integer(0)
  matched_c <- integer(0)
  usados    <- integer(0)

  for (ti in idx_t) {
    pool <- if (replacement) idx_c else setdiff(idx_c, usados)
    if (length(pool) == 0) break

    distancias <- abs(ps[ti] - ps[pool])
    mejor_pos  <- which.min(distancias)
    mejor_idx  <- pool[mejor_pos]
    mejor_dist <- distancias[mejor_pos]

    if (!is.null(caliper) && mejor_dist >= caliper) next

    matched_t <- c(matched_t, ti)
    matched_c <- c(matched_c, mejor_idx)
    if (!replacement) usados <- c(usados, mejor_idx)
  }

  if (length(matched_t) == 0) {
    message("Matching -- ningun par encontrado. Considera ampliar el caliper o usar replacement=TRUE.")
    return(NULL)
  }

  rows_t <- df[matched_t, ]
  rows_c <- df[matched_c, ]
  rows_t$pair_id <- seq_len(nrow(rows_t))
  rows_c$pair_id <- seq_len(nrow(rows_c))

  resultado <- rbind(rows_t, rows_c)
  rownames(resultado) <- NULL

  message(sprintf(
    "Matching 1:1 -- %d pares  (tratados=%d, controles=%d)",
    nrow(rows_t), nrow(rows_t), nrow(rows_c)
  ))
  return(resultado)
}


#' @keywords internal
.balance_subclassification <- function(df, tratamiento, col_ps, n_subclases) {
  df_out <- df

  ps <- df_out[[col_ps]]
  # qcut con manejo de duplicados
  breaks <- unique(stats::quantile(ps, probs = seq(0, 1, length.out = n_subclases + 1)))
  if (length(breaks) < 3) {
    breaks <- seq(min(ps), max(ps), length.out = n_subclases + 1)
  }
  # include.lowest para que el mínimo quede incluido
  df_out$subclase <- as.integer(cut(ps, breaks = breaks,
                                     labels = FALSE, include.lowest = TRUE))

  # pesos = proporción de la subclase en la muestra total
  tab   <- table(df_out$subclase)
  props <- prop.table(tab)
  df_out$peso_subclase <- as.numeric(props[as.character(df_out$subclase)])

  n_subclases_real <- length(unique(df_out$subclase[!is.na(df_out$subclase)]))
  message(sprintf(
    "Subclasificacion -- %d subclases (solicitadas: %d)",
    n_subclases_real, n_subclases
  ))

  for (sc in sort(unique(df_out$subclase[!is.na(df_out$subclase)]))) {
    sub <- df_out[!is.na(df_out$subclase) & df_out$subclase == sc, ]
    nt  <- sum(sub[[tratamiento]] == 1, na.rm = TRUE)
    nc  <- sum(sub[[tratamiento]] == 0, na.rm = TRUE)
    message(sprintf(
      "  Subclase %2d: n=%4d  tratados=%3d  controles=%3d  PS_medio=%.3f",
      sc, nrow(sub), nt, nc, mean(sub[[col_ps]])
    ))
  }

  return(df_out)
}


#' @keywords internal
.balance_trimming <- function(df, tratamiento, col_ps, kappa) {
  mascara  <- df[[col_ps]] > kappa & df[[col_ps]] < (1 - kappa)
  n_drop   <- sum(!mascara)
  df_trim  <- df[mascara, ]
  rownames(df_trim) <- NULL

  pct <- round(n_drop / nrow(df) * 100, 1)
  message(sprintf(
    "Trimming (kappa=%.2f) -- %d unidades eliminadas (%.1f%%)  -> n final=%d  (tratados=%d, controles=%d)",
    kappa, n_drop, pct, nrow(df_trim),
    sum(df_trim[[tratamiento]] == 1, na.rm = TRUE),
    sum(df_trim[[tratamiento]] == 0, na.rm = TRUE)
  ))
  message(sprintf(
    "  Rango PS conservado: [%.4f, %.4f]",
    min(df_trim[[col_ps]]), max(df_trim[[col_ps]])
  ))

  return(df_trim)
}


#' @keywords internal
.balance_truncating <- function(df, tratamiento, col_ps, kappa) {
  df_out <- df
  ps_original <- df_out[[col_ps]]

  df_out[[paste0(col_ps, "_original")]] <- ps_original
  df_out[[col_ps]] <- pmax(kappa, pmin(ps_original, 1 - kappa))

  n_truncados <- sum(ps_original < kappa | ps_original > (1 - kappa))
  pct <- round(n_truncados / nrow(df) * 100, 1)

  message(sprintf(
    "Truncating (kappa=%.2f) -- %d PS truncados (%.1f%%)  -> n=%d (sin eliminar filas)",
    kappa, n_truncados, pct, nrow(df_out)
  ))
  message(sprintf(
    "  PS original:  [%.4f, %.4f]  ->  PS truncado: [%.4f, %.4f]",
    min(ps_original), max(ps_original),
    min(df_out[[col_ps]]), max(df_out[[col_ps]])
  ))

  return(df_out)
}
