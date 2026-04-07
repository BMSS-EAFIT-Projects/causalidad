"""
balance.py
==========
Función balance() — balanceo por propensity score.

Lee el PS desde una columna del DataFrame (generada por propensity_score()).
No estima el PS internamente.

Métodos disponibles (fiel a las slides de LaLonde / Taddeo 2025):
  · 'matching'    → matching 1:1 vecino más cercano en PS
  · 'subclassif'  → subclasificación por quintiles de PS
  · 'trimming'    → eliminar unidades con PS < kappa o PS > 1-kappa  [slide 65]
  · 'truncating'  → recortar PS a max{kappa, min{PS, 1-kappa}}       [slide 65]

Uso
---
    from propensity_score import propensity_score
    from balance import balance

    df     = propensity_score(df, tratamiento='treatment', covariables=covs)
    df_bal = balance(df, tratamiento='treatment', metodo='trimming')
"""

import pandas as pd
import numpy as np


# ══════════════════════════════════════════════════════════════
# MÉTODO 1: MATCHING 1:1
# ══════════════════════════════════════════════════════════════

def _matching(df, tratamiento, col_ps, replacement, caliper):
    """
    Matching 1:1 por vecino más cercano en el propensity score.

    Para cada unidad tratada i:
        j* = argmin_{j en controles} |PS_i - PS_j|

    Si caliper no es None, descarta el par si la distancia >= caliper.

    Columnas modificadas/añadidas
    ------------------------------
    {col_ps}  : PS original (sin cambios).
    pair_id   : Entero que identifica el par tratado-control.
    """
    idx_t = df.index[df[tratamiento] == 1].tolist()
    idx_c = df.index[df[tratamiento] == 0].tolist()

    ps_s = df[col_ps]
    matched_t, matched_c = [], []
    usados = set() if not replacement else None

    for ti in idx_t:
        pool = [j for j in idx_c if (replacement or j not in usados)]
        if not pool:
            break
        distancias = {j: abs(ps_s[ti] - ps_s[j]) for j in pool}
        mejor      = min(distancias, key=distancias.get)
        if caliper is not None and distancias[mejor] >= caliper:
            continue
        matched_t.append(ti)
        matched_c.append(mejor)
        if not replacement:
            usados.add(mejor)

    if not matched_t:
        print("Matching — ningún par encontrado. "
              "Considera ampliar el caliper o usar replacement=True.")
        return None

    rows_t = df.loc[matched_t].copy()
    rows_c = df.loc[matched_c].copy()
    rows_t['pair_id'] = np.arange(len(rows_t))
    rows_c['pair_id'] = np.arange(len(rows_c))

    resultado = pd.concat([rows_t, rows_c], ignore_index=True)
    print(f"Matching 1:1 — {len(rows_t)} pares  "
          f"(tratados={len(rows_t)}, controles={len(rows_c)})")
    return resultado


# ══════════════════════════════════════════════════════════════
# MÉTODO 2: SUBCLASIFICACIÓN
# ══════════════════════════════════════════════════════════════

def _subclassification(df, tratamiento, col_ps, n_subclases):
    """
    Subclasificación por quintiles (o n_subclases) del PS.

    Columnas añadidas
    -----------------
    subclase      : Entero 1..n_subclases que identifica la subclase.
    peso_subclase : Proporción de la subclase en la muestra total.
    """
    df = df.copy()

    try:
        df['subclase'] = pd.qcut(df[col_ps], q=n_subclases,
                                 labels=False, duplicates='drop') + 1
    except ValueError:
        df['subclase'] = pd.cut(df[col_ps], bins=n_subclases,
                                labels=False) + 1

    pesos = df['subclase'].value_counts(normalize=True).sort_index()
    df['peso_subclase'] = df['subclase'].map(pesos)

    print(f"Subclasificación — {df['subclase'].nunique()} subclases "
          f"(solicitadas: {n_subclases})")
    for sc in sorted(df['subclase'].dropna().unique()):
        sub = df[df['subclase'] == sc]
        nt  = (sub[tratamiento] == 1).sum()
        nc  = (sub[tratamiento] == 0).sum()
        print(f"  Subclase {int(sc):2d}: n={len(sub):4d}  "
              f"tratados={nt:3d}  controles={nc:3d}  "
              f"PS_medio={sub[col_ps].mean():.3f}")
    return df


# ══════════════════════════════════════════════════════════════
# MÉTODO 3: TRIMMING  (slides p.65)
# ══════════════════════════════════════════════════════════════

def _trimming(df, tratamiento, col_ps, kappa):
    """
    Trimming — eliminar unidades con PS fuera del soporte común.

    Fórmula exacta de los slides (p.65):
        DROP UNITS with PS_i < kappa  or  PS_i > 1-kappa

    Ejemplo LaLonde de los slides (p.72), kappa=0.05:
        lalonde_trim <- lalonde[PS > Q & PS < (1-Q), ]

    El DataFrame de salida tiene menos filas que el de entrada.
    La columna del PS no se modifica.
    """
    mascara = (df[col_ps] > kappa) & (df[col_ps] < 1 - kappa)
    n_drop  = (~mascara).sum()
    df_trim = df[mascara].copy()

    pct = n_drop / len(df) * 100
    print(f"Trimming (kappa={kappa}) — "
          f"{n_drop} unidades eliminadas ({pct:.1f}%)  "
          f"-> n final={len(df_trim)}  "
          f"(tratados={int((df_trim[tratamiento]==1).sum())}, "
          f"controles={int((df_trim[tratamiento]==0).sum())})")
    print(f"  Rango PS conservado: "
          f"[{df_trim[col_ps].min():.4f}, {df_trim[col_ps].max():.4f}]")
    return df_trim


# ══════════════════════════════════════════════════════════════
# MÉTODO 4: TRUNCATING  (slides p.65)
# ══════════════════════════════════════════════════════════════

def _truncating(df, tratamiento, col_ps, kappa):
    """
    Truncating — recortar el PS a [kappa, 1-kappa] sin eliminar filas.

    Fórmula exacta de los slides (p.65):
        PS_tilde_i = max{kappa, min{PS_i, 1-kappa}}

    Columnas modificadas/añadidas
    ------------------------------
    {col_ps}_original : PS original antes de truncar.
    {col_ps}          : PS truncado (sobreescribe la columna original).
    """
    df = df.copy()
    ps_original = df[col_ps].values

    df[f'{col_ps}_original'] = ps_original
    df[col_ps]               = np.clip(ps_original, kappa, 1 - kappa)

    n_truncados = ((ps_original < kappa) | (ps_original > 1 - kappa)).sum()
    pct = n_truncados / len(df) * 100

    print(f"Truncating (kappa={kappa}) — "
          f"{n_truncados} PS truncados ({pct:.1f}%)  "
          f"-> n={len(df)} (sin eliminar filas)")
    print(f"  PS original:  [{ps_original.min():.4f}, {ps_original.max():.4f}]  ->  "
          f"PS truncado: [{df[col_ps].min():.4f}, {df[col_ps].max():.4f}]")
    return df


# ══════════════════════════════════════════════════════════════
# FUNCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════

def balance(df, tratamiento,
            metodo='trimming',
            col_ps='propensity_score',
            kappa=0.05,
            n_subclases=5,
            replacement=False,
            caliper=None):
    """
    Balancea una muestra observacional usando el propensity score
    almacenado en una columna del DataFrame.

    Requiere haber llamado antes a propensity_score(), que añade
    la columna col_ps al DataFrame.

    Parámetros
    ----------
    df          : pandas DataFrame con la columna del PS ya calculada.
    tratamiento : Nombre de la columna de tratamiento (valores 0/1).
    metodo      : Método de balanceo:
                    'matching'    -> matching 1:1 vecino más cercano
                    'subclassif'  -> subclasificación por quintiles
                    'trimming'    -> eliminar unidades con PS < kappa o > 1-kappa
                    'truncating'  -> recortar PS a [kappa, 1-kappa]
    col_ps      : Nombre de la columna del PS (default 'propensity_score').
    kappa       : Umbral para trimming y truncating (default 0.05).
                  Rango válido: (0, 0.5). Slides recomiendan 0.05-0.10.
    n_subclases : Número de subclases para 'subclassif' (default 5).
    replacement : Para 'matching': permitir reemplazo (default False).
    caliper     : Para 'matching': distancia máxima aceptable en PS
                  (default None = sin restricción).

    Retorna
    -------
    pd.DataFrame balanceado con las columnas originales más:

      Todos los métodos:
        · {col_ps}                      : PS (original o truncado)

      Solo 'matching':
        · pair_id                       : ID del par tratado-control

      Solo 'subclassif':
        · subclase                      : Subclase asignada (1..n_subclases)
        · peso_subclase                 : Proporción de la subclase

      Solo 'truncating':
        · {col_ps}_original             : PS antes de truncar

    Retorna None si el proceso falla.
    """
    # ── Validaciones ──────────────────────────────────────────
    if col_ps not in df.columns:
        raise ValueError(
            f"La columna '{col_ps}' no existe en df. "
            f"Llama primero a propensity_score()."
        )

    metodos_validos = ('matching', 'subclassif', 'trimming', 'truncating')
    if metodo not in metodos_validos:
        raise ValueError(f"metodo='{metodo}' no reconocido. "
                         f"Opciones: {metodos_validos}")

    if metodo in ('trimming', 'truncating') and not (0 < kappa < 0.5):
        raise ValueError(f"kappa debe estar en (0, 0.5). Recibido: {kappa}")

    if df[tratamiento].nunique() < 2:
        print("balance() — Error: sin variación en el tratamiento.")
        return None

    # Descartar filas sin PS (pueden venir de NaN en covariables)
    df_clean = df.dropna(subset=[col_ps]).copy()
    n_drop   = len(df) - len(df_clean)
    if n_drop > 0:
        print(f"balance() — {n_drop} filas sin PS descartadas.")

    # ── Encabezado ────────────────────────────────────────────
    n_t = int((df_clean[tratamiento] == 1).sum())
    n_c = int((df_clean[tratamiento] == 0).sum())
    print(f"\n{'─'*50}")
    print(f"  balance() | metodo='{metodo}' | "
          f"n={len(df_clean)} (T={n_t}, C={n_c})")
    print(f"{'─'*50}")

    # ── Método ───────────────────────────────────────────────
    if metodo == 'matching':
        resultado = _matching(df_clean, tratamiento, col_ps,
                              replacement=replacement, caliper=caliper)
    elif metodo == 'subclassif':
        resultado = _subclassification(df_clean, tratamiento, col_ps,
                                       n_subclases=n_subclases)
    elif metodo == 'trimming':
        resultado = _trimming(df_clean, tratamiento, col_ps, kappa=kappa)
    elif metodo == 'truncating':
        resultado = _truncating(df_clean, tratamiento, col_ps, kappa=kappa)

    print(f"{'─'*50}\n")
    return resultado