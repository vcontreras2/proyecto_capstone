"""
Datos de la instancia (Anexo A de formulacion_compacta.pdf).

Datos reales cargados desde "Anexo a Datos.pdf" (instancia de referencia).
"""

# ---------------------------------------------------------------------------
# 1) Arbol de escenarios: nodo -> nodo padre (la raiz tiene padre None)
# ---------------------------------------------------------------------------
PADRE = {1: None, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 3, 8: 3, 9: 3, 10: 4, 11: 4}

# ---------------------------------------------------------------------------
# 2) Probabilidad pi_n de cada nodo (producto de condicionales desde la raiz)
# ---------------------------------------------------------------------------
_COND = {1: 1.0, 2: 0.3, 3: 0.4, 4: 0.3, 5: 0.7, 6: 0.3,
         7: 1 / 3, 8: 1 / 3, 9: 1 / 3, 10: 0.6, 11: 0.4}   # prob. condicional (Anexo A, sec. 5)


def _prob(n):
    return _COND[n] * (_prob(PADRE[n]) if PADRE[n] else 1.0)


PROB = {n: _prob(n) for n in PADRE}

# ---------------------------------------------------------------------------
# 3) Demanda D[(vino, etiqueta, nodo)] en botellas (Anexo A, sec. 5)
# ---------------------------------------------------------------------------
ETIQUETAS = {1: [1, 2, 3], 2: [1, 2]}      # J_i

_TABLA = {   # producto: demanda en los nodos N1..N11
    (1, 1): [23525, 53218, 34328, 18118, 81856, 45302, 50426, 31482, 12538, 30856, 10426],
    (1, 2): [24328, 15184, 28525, 45832, 24546, 31371, 23569, 40767, 57965, 20546, 45569],
    (1, 3): [20368, 38150, 21080, 6762, 4920, 25765, 35043, 25646, 16248, 10920, 11043],
    (2, 1): [21080, 39387, 30368, 12646, 34115, 22773, 37045, 27831, 18616, 24115, 17045],
    (2, 2): [22991, 38048, 27991, 14949, 33440, 22824, 35085, 25147, 15209, 23440, 15085],
}
DEM = {(i, j, n): _TABLA[(i, j)][n - 1] for (i, j) in _TABLA for n in PADRE}

# ---------------------------------------------------------------------------
# 4) Parametros del modelo (Tabla 2). Normalmente no hay que tocarlos.
# ---------------------------------------------------------------------------
PARAMS = dict(
    T_prod=84.0,                       # horas de linea por periodo
    K_tank=20, V_tank=10_000.0,        # estanques por periodo, litros por estanque
    V_bot=0.75, U=1.0,                 # litros por botella, fraccion max de estanque sin usar
    Tz_b=1.5, Tz_bl=1.5, Tz_l=0.5,     # horas de set-up
    Tw_b=0.00014, Tw_bl=0.00028, Tw_l=0.00028,   # horas por botella
    Cz_b=1500.0, Cz_bl=1500.0, Cz_l=500.0,       # costo set-up ($)
    Cs_b=10.0, Cs_bl=15.0,             # costo guardar sin etiqueta / terminada ($ por periodo)
    Cb_bl=15_000.0,                    # costo botella atrasada ($ por periodo)
    M_b=266_667.0, M_bl=266_667.0, M_l=298_214.0,  # maximos de produccion
)
