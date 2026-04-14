# ── Datos con ATE conocido = 2.0 ──────────────────────────────────────────
make_rct <- function(seed = 99, n = 500) {
  set.seed(seed)
  x1 <- rnorm(n); x2 <- rnorm(n)
  tr <- rbinom(n, 1, 0.5)          # RCT: asignacion aleatoria
  y  <- 2 * tr + x1 + rnorm(n)    # ATE verdadero = 2
  df <- data.frame(y = y, tr = tr, x1 = x1, x2 = x2)
  propensity_score(df, "tr", c("x1", "x2"))
}

make_obs <- function(seed = 42, n = 800) {
  set.seed(seed)
  x1 <- rnorm(n); x2 <- rnorm(n)
  tr <- rbinom(n, 1, plogis(0.5 * x1 - 0.3 * x2))
  y  <- 2 * tr + x1 + rnorm(n)
  df <- data.frame(y = y, tr = tr, x1 = x1, x2 = x2)
  propensity_score(df, "tr", c("x1", "x2"))
}

# ── Estructura del output ─────────────────────────────────────────────────
test_that("calcular_ate retorna lista con todas las claves esperadas", {
  df  <- make_rct()
  res <- calcular_ate(df, "y", "tr", c("x1", "x2"))

  claves_esperadas <- c("n_tratados", "n_controles",
                        "naive",
                        "regresion", "regresion_ic95", "regresion_pvalor",
                        "g_formula",
                        "ht", "hajek",
                        "msm", "msm_ic95", "msm_pvalor",
                        "dr")
  expect_true(all(claves_esperadas %in% names(res)))
})

# ── Estimadores en RCT (deben estar cerca del ATE = 2) ───────────────────
test_that("todos los estimadores convergen al ATE verdadero en RCT", {
  df  <- make_rct(n = 1000)
  res <- calcular_ate(df, "y", "tr", c("x1", "x2"))

  tolerancia <- 0.4   # margen amplio para n razonable
  expect_equal(res$naive,     2, tolerance = tolerancia)
  expect_equal(res$regresion, 2, tolerance = tolerancia)
  expect_equal(res$g_formula, 2, tolerance = tolerancia)
  expect_equal(res$ht,        2, tolerance = tolerancia)
  expect_equal(res$hajek,     2, tolerance = tolerancia)
  expect_equal(res$msm,       2, tolerance = tolerancia)
  expect_equal(res$dr,        2, tolerance = tolerancia)
})

# ── IC del 95 % incluye el ATE verdadero ──────────────────────────────────
test_that("IC de regresion y MSM incluyen el ATE verdadero", {
  df  <- make_rct(n = 1000)
  res <- calcular_ate(df, "y", "tr", c("x1", "x2"))

  expect_true(res$regresion_ic95[1] < 2 && 2 < res$regresion_ic95[2])
  expect_true(res$msm_ic95[1]       < 2 && 2 < res$msm_ic95[2])
})

# ── DR es robusto cuando el PS es correcto ────────────────────────────────
test_that("DR es consistente con datos observacionales y PS correcto", {
  df_ps   <- make_obs(n = 1200)
  df_trim <- balance(df_ps, "tr", metodo = "trimming", kappa = 0.05)
  res     <- calcular_ate(df_trim, "y", "tr", c("x1", "x2"))

  # ATE verdadero = 2; tolerancia mas generosa por confusión residual
  expect_equal(res$dr,        2, tolerance = 0.5)
  expect_equal(res$hajek,     2, tolerance = 0.5)
  expect_equal(res$regresion, 2, tolerance = 0.5)
})

# ── Naive esta sesgado en datos observacionales ────────────────────────────
test_that("naive esta sesgado en datos observacionales", {
  # Con confusión fuerte, naive NO coincide con los otros estimadores
  set.seed(10)
  n  <- 1000
  x1 <- rnorm(n)
  tr <- rbinom(n, 1, plogis(2 * x1))  # confusión fuerte
  y  <- 2 * tr + 3 * x1 + rnorm(n)   # ATE = 2 pero x1 confunde
  df <- data.frame(y = y, tr = tr, x1 = x1)
  df <- propensity_score(df, "tr", "x1")

  res <- calcular_ate(df, "y", "tr", "x1")

  # naive deberia diferir del DR en más de 0.5 bajo confusión fuerte
  expect_true(abs(res$naive - res$dr) > 0.5)
})

# ── Errores esperados ─────────────────────────────────────────────────────
test_that("calcular_ate lanza error si falta columna de PS", {
  df <- data.frame(y = rnorm(50), tr = rbinom(50, 1, 0.5), x1 = rnorm(50))
  expect_error(calcular_ate(df, "y", "tr", "x1"), "propensity_score")
})

test_that("calcular_ate lanza error con columna faltante", {
  df  <- make_rct(n = 100)
  expect_error(calcular_ate(df, "y", "tr", c("x1", "no_existe")), "no encontradas")
})
