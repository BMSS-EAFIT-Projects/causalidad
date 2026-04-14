# causality <img src="https://img.shields.io/badge/R-package-blue"/>

Paquete R de inferencia causal.

## Instalación

```r
# Desde la carpeta del paquete:
devtools::install("paqueteR/causality")

# O directamente con devtools desde GitHub (cuando esté publicado):
# devtools::install_github("afpuerta/causality")
```

## Flujo de trabajo

```r
library(causality)

# 1. Datos (ejemplo con LaLonde)
data(lalonde, package = "MatchIt")
covs <- c("age", "educ", "black", "hispan", "married", "nodegree", "re74", "re75")

# 2. Estimar Propensity Score
df <- propensity_score(lalonde, tratamiento = "treat", covariables = covs)

# 3. Balancear (trimming por defecto, kappa = 0.05)
df_bal <- balance(df, tratamiento = "treat", metodo = "trimming")

# 4. Calcular ATE con todos los estimadores
res <- calcular_ate(df_bal, resultado = "re78",
                    tratamiento = "treat", covariables = covs)

# 5. Visualizar balance
tabla <- visualizar_balance(df_bal, "treat", covs)
print(tabla)
```

## Estimadores disponibles en `calcular_ate()`

| Clave             | Estimador                                  | Slide |
|-------------------|--------------------------------------------|-------|
| `naive`           | Diferencia de medias                        | 5/9   |
| `regresion`       | OLS: Y ~ A + X                             | 5/9   |
| `g_formula`       | G-fórmula / Estandarización                | 87    |
| `ht`              | Horvitz-Thompson (IPW no normalizado)      | 54    |
| `hajek`           | Hajek (IPW normalizado)                    | 65    |
| `msm`             | Modelo Estructural Marginal (WLS)          | 77    |
| `dr`              | Doblemente Robusto (AIPW)                  | 78    |

## Métodos de balanceo en `balance()`

| `metodo`       | Descripción                                      | Slide |
|----------------|--------------------------------------------------|-------|
| `"trimming"`   | Eliminar unidades con PS fuera de [κ, 1-κ]      | 65    |
| `"truncating"` | Recortar PS a [κ, 1-κ] sin eliminar filas        | 65    |
| `"matching"`   | Matching 1:1 vecino más cercano                  | —     |
| `"subclassif"` | Subclasificación por quintiles de PS             | —     |

## Correr los tests

```r
devtools::test("paqueteR/causality")
# o desde la carpeta del paquete:
# testthat::test_dir("tests/testthat")
```
