# ── Datos compartidos ─────────────────────────────────────────────────────
make_df <- function(seed = 7) {
  set.seed(seed)
  n  <- 300
  x1 <- rnorm(n); x2 <- rnorm(n)
  tr <- rbinom(n, 1, plogis(0.4 * x1 - 0.3 * x2))
  y  <- 1.5 * tr + 0.8 * x1 + rnorm(n)
  df <- data.frame(y = y, tr = tr, x1 = x1, x2 = x2)
  propensity_score(df, "tr", c("x1", "x2"))
}

# ── trimming ──────────────────────────────────────────────────────────────
test_that("trimming elimina filas con PS extremo", {
  df_ps   <- make_df()
  df_trim <- balance(df_ps, "tr", metodo = "trimming", kappa = 0.05)

  expect_true(nrow(df_trim) < nrow(df_ps))
  expect_true(all(df_trim$propensity_score > 0.05))
  expect_true(all(df_trim$propensity_score < 0.95))
})

test_that("trimming lanza error con kappa invalido", {
  df_ps <- make_df()
  expect_error(balance(df_ps, "tr", metodo = "trimming", kappa = 0.6), "kappa")
  expect_error(balance(df_ps, "tr", metodo = "trimming", kappa = 0),   "kappa")
})

# ── truncating ────────────────────────────────────────────────────────────
test_that("truncating conserva el numero de filas", {
  df_ps   <- make_df()
  df_trunc <- balance(df_ps, "tr", metodo = "truncating", kappa = 0.05)

  expect_equal(nrow(df_trunc), nrow(df_ps))
  expect_true("propensity_score_original" %in% names(df_trunc))
  expect_true(all(df_trunc$propensity_score >= 0.05))
  expect_true(all(df_trunc$propensity_score <= 0.95))
})

# ── subclassif ────────────────────────────────────────────────────────────
test_that("subclassif crea columnas subclase y peso_subclase", {
  df_ps  <- make_df()
  df_sub <- balance(df_ps, "tr", metodo = "subclassif", n_subclases = 5)

  expect_true("subclase" %in% names(df_sub))
  expect_true("peso_subclase" %in% names(df_sub))
  expect_true(length(unique(df_sub$subclase)) <= 5)
  expect_true(all(df_sub$peso_subclase > 0))
})

# ── matching ──────────────────────────────────────────────────────────────
test_that("matching genera pares tratado-control con pair_id", {
  df_ps    <- make_df()
  df_match <- balance(df_ps, "tr", metodo = "matching", replacement = FALSE)

  expect_true("pair_id" %in% names(df_match))
  n_t <- sum(df_match$tr == 1)
  n_c <- sum(df_match$tr == 0)
  expect_equal(n_t, n_c)   # 1:1
})

test_that("balance lanza error si falta columna de PS", {
  df <- data.frame(tr = rbinom(50, 1, 0.5), x = rnorm(50))
  expect_error(balance(df, "tr"), "propensity_score")
})

test_that("balance lanza error con metodo no reconocido", {
  df_ps <- make_df()
  expect_error(balance(df_ps, "tr", metodo = "inventado"), "no reconocido")
})
