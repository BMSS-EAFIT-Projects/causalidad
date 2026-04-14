test_that("propensity_score agrega columna con rango correcto", {
  set.seed(42)
  n  <- 200
  x1 <- rnorm(n); x2 <- rnorm(n)
  lp <- 0.3 * x1 - 0.5 * x2
  tr <- rbinom(n, 1, plogis(lp))
  y  <- 2 * tr + x1 + rnorm(n)
  df <- data.frame(y = y, tr = tr, x1 = x1, x2 = x2)

  df_ps <- propensity_score(df, tratamiento = "tr", covariables = c("x1", "x2"))

  expect_true("propensity_score" %in% names(df_ps))
  expect_true(all(df_ps$propensity_score > 0, na.rm = TRUE))
  expect_true(all(df_ps$propensity_score < 1, na.rm = TRUE))
  expect_equal(nrow(df_ps), nrow(df))
})

test_that("propensity_score respeta nombre de columna personalizado", {
  set.seed(1)
  df <- data.frame(
    trt = rbinom(100, 1, 0.4),
    x1  = rnorm(100)
  )
  df_ps <- propensity_score(df, "trt", "x1", columna = "mi_ps")
  expect_true("mi_ps" %in% names(df_ps))
  expect_false("propensity_score" %in% names(df_ps))
})

test_that("propensity_score lanza error con columna faltante", {
  df <- data.frame(trt = rbinom(50, 1, 0.5), x1 = rnorm(50))
  expect_error(
    propensity_score(df, "trt", c("x1", "x_no_existe")),
    "no encontradas"
  )
})

test_that("propensity_score lanza error sin variacion en tratamiento", {
  df <- data.frame(trt = rep(1, 50), x1 = rnorm(50))
  expect_error(
    propensity_score(df, "trt", "x1"),
    "variacion"
  )
})
