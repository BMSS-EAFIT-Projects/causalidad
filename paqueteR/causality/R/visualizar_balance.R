#' Visualizar el Balance de Covariables
#'
#' Genera una tabla de balance estadístico y gráficos de densidad y boxplots
#' para cada covariable, comparando tratados y controles.
#'
#' El balance se define con base en la prueba t de igualdad de medias
#' (p-valor > 0.05), consistente con el criterio de los slides.
#' También se reporta el Standardized Mean Difference (SMD) como métrica
#' complementaria.
#'
#' @param df Data frame (puede ser el balanceado con [balance()]).
#' @param tratamiento Nombre de la columna de tratamiento (valores 0/1).
#' @param covariables Vector de nombres de covariables a evaluar.
#' @param mostrar_graficos Lógico. Si `TRUE` (default), muestra los gráficos.
#'
#' @return Data frame con columnas:
#'   - `Covariable`
#'   - `Tratados (Media ± SD)`
#'   - `Controles (Media ± SD)`
#'   - `p_valor` (prueba t de Welch)
#'   - `SMD` (Standardized Mean Difference)
#'   - `Balanceado` ("Sí" si p-valor > 0.05)
#'
#' @examples
#' data(lalonde, package = "MatchIt")
#' covs <- c("age", "educ", "black", "hispan", "married", "nodegree",
#'           "re74", "re75")
#' df      <- propensity_score(lalonde, "treat", covs)
#' df_trim <- balance(df, "treat", metodo = "trimming")
#' tabla   <- visualizar_balance(df_trim, "treat", covs)
#' print(tabla)
#'
#' @export
visualizar_balance <- function(df, tratamiento, covariables,
                                mostrar_graficos = TRUE) {

  df_clean  <- df[stats::complete.cases(df[, c(tratamiento, covariables)]), ]
  tratados  <- df_clean[df_clean[[tratamiento]] == 1, ]
  controles <- df_clean[df_clean[[tratamiento]] == 0, ]

  resultados <- vector("list", length(covariables))

  for (i in seq_along(covariables)) {
    cov <- covariables[i]

    x_t <- tratados[[cov]]
    x_c <- controles[[cov]]

    mean_t <- mean(x_t, na.rm = TRUE)
    std_t  <- stats::sd(x_t, na.rm = TRUE)
    mean_c <- mean(x_c, na.rm = TRUE)
    std_c  <- stats::sd(x_c, na.rm = TRUE)

    # Prueba t de Welch (igual_var = FALSE como en Python)
    ttest  <- stats::t.test(x_t, x_c, var.equal = FALSE)
    p_val  <- ttest$p.value

    # SMD
    var_pooled <- (stats::var(x_t, na.rm = TRUE) +
                   stats::var(x_c, na.rm = TRUE)) / 2
    std_pooled <- sqrt(var_pooled)
    smd <- if (std_pooled != 0) abs((mean_t - mean_c) / std_pooled) else 0

    resultados[[i]] <- data.frame(
      Covariable                  = cov,
      `Tratados (Media ± SD)`     = sprintf("%.2f ± %.2f", mean_t, std_t),
      `Controles (Media ± SD)`    = sprintf("%.2f ± %.2f", mean_c, std_c),
      `p_valor (Prueba de Medias)`= round(p_val, 4),
      SMD                         = round(smd, 4),
      Balanceado                  = ifelse(p_val > 0.05, "Sí", "No"),
      check.names                 = FALSE,
      stringsAsFactors            = FALSE
    )
  }

  tabla <- do.call(rbind, resultados)

  # ── Gráficos ──────────────────────────────────────────────────────────────
  if (mostrar_graficos && requireNamespace("ggplot2", quietly = TRUE)) {
    df_long <- do.call(rbind, lapply(covariables, function(cov) {
      data.frame(
        covariable  = cov,
        valor       = df_clean[[cov]],
        tratamiento = factor(df_clean[[tratamiento]],
                             levels = c(0, 1),
                             labels = c("Control", "Tratado")),
        stringsAsFactors = FALSE
      )
    }))

    p_dens <- ggplot2::ggplot(
      df_long,
      ggplot2::aes(x = valor, fill = tratamiento, color = tratamiento)
    ) +
      ggplot2::geom_density(alpha = 0.4) +
      ggplot2::facet_wrap(~ covariable, scales = "free") +
      ggplot2::labs(title = "Distribuciones por grupo",
                    x = NULL, y = "Densidad",
                    fill = "Grupo", color = "Grupo") +
      ggplot2::theme_bw() +
      ggplot2::scale_fill_manual(values  = c("Control" = "#2196F3",
                                              "Tratado" = "#F44336")) +
      ggplot2::scale_color_manual(values = c("Control" = "#2196F3",
                                              "Tratado" = "#F44336"))

    p_box <- ggplot2::ggplot(
      df_long,
      ggplot2::aes(x = tratamiento, y = valor, fill = tratamiento)
    ) +
      ggplot2::geom_boxplot(alpha = 0.6, outlier.size = 0.8) +
      ggplot2::facet_wrap(~ covariable, scales = "free_y") +
      ggplot2::labs(title = "Boxplots por grupo",
                    x = NULL, y = NULL) +
      ggplot2::theme_bw() +
      ggplot2::scale_fill_manual(values = c("Control" = "#2196F3",
                                             "Tratado" = "#F44336"))

    print(p_dens)
    print(p_box)
  }

  return(tabla)
}
