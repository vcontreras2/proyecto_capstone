"""
Modelo de embotellado/etiquetado con postergacion (formulacion_compacta.pdf).
Se resuelve con PuLP + HiGHS (gratis, sin Gurobi).

Uso rapido:
    import modelo
    res = modelo.resolver(PL=2)
    modelo.graficar_arbol(res); modelo.graficar_decisiones(res)
    tabla = modelo.comparar_politicas(); modelo.graficar_comparacion(tabla)
"""
import time

import pulp
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

import datos


# ---------------------------------------------------------------------------
# Construccion y resolucion
# ---------------------------------------------------------------------------
def resolver(PL=2, contra_pedido=False, datos_mod=datos, tiempo_max=300, verbose=False, **override):
    """Construye y resuelve el modelo. PL=0 -> a stock; PL>=2 -> hibrida.
    contra_pedido=True agrega R15 (sbl = 0) y fuerza PL=0.
    `override` permite cambiar parametros de la Tabla 2, ej. Cb_bl=1000, T_prod=100."""
    P = {**datos_mod.PARAMS, **override}
    padre, pi, D, J = datos_mod.PADRE, datos_mod.PROB, datos_mod.DEM, datos_mod.ETIQUETAS
    if contra_pedido:
        PL = 0
    N = sorted(padre)
    raiz = [n for n in N if padre[n] is None][0]
    hijos = {n: [m for m in N if padre[m] == n] for n in N}
    L = [n for n in N if not hijos[n]]            # ultimo periodo
    I = sorted(J)
    IJ = [(i, j) for i in I for j in J[i]]
    AUX = 0                                         # nodo ficticio naux
    a = {**padre, raiz: AUX}
    Tplus = [AUX] + N

    t0 = time.perf_counter()
    m = pulp.LpProblem("postergacion", pulp.LpMinimize)
    def V(nombre, idx, low=None, up=None, cat="Continuous"):
        return {k: pulp.LpVariable(f"{nombre}_" + "_".join(map(str, k if isinstance(k, tuple) else (k,))),
                                   lowBound=low, upBound=up, cat=cat) for k in idx}
    wbl = V("wbl", [(i, j, n) for i, j in IJ for n in N], low=0)
    wb = V("wb", [(i, n) for i in I for n in N], low=0)
    wl = V("wl", [(i, j, n) for i, j in IJ for n in N], low=0)
    zbl = V("zbl", [(i, j, n) for i, j in IJ for n in N], cat="Binary")
    zb = V("zb", [(i, n) for i in I for n in N], cat="Binary")
    zl = V("zl", [(i, j, n) for i, j in IJ for n in N], cat="Binary")
    sb = V("sb", [(i, n) for i in I for n in N], low=0)
    sbl = V("sbl", [(i, j, n) for i, j in IJ for n in N], low=0)
    bbl = V("bbl", [(i, j, n) for i, j in IJ for n in N], low=0)
    y = V("y", [(i, n) for i in I for n in N], low=0, cat="Integer")
    u = V("u", [(i, n) for i in I for n in N], low=0)
    zpos = V("zpos", IJ, cat="Binary")

    # Valores del nodo ficticio (R6)
    W0 = {(i, j): D[(i, j, raiz)] for i, j in IJ}
    gwb = lambda i, n: 0 if n == AUX else wb[i, n]
    gwbl = lambda i, j, n: W0[i, j] if n == AUX else wbl[i, j, n]
    gsb = lambda i, n: 0 if n == AUX else sb[i, n]
    gsbl = lambda i, j, n: 0 if n == AUX else sbl[i, j, n]
    gbbl = lambda i, j, n: 0 if n == AUX else bbl[i, j, n]

    # Objetivo (FO)
    m += pulp.lpSum(
        pi[n] * (
            pulp.lpSum(P["Cz_b"] * zb[i, n] + P["Cs_b"] * sb[i, n] for i in I)
            + pulp.lpSum(P["Cz_bl"] * zbl[i, j, n] + P["Cz_l"] * zl[i, j, n]
                         + P["Cs_bl"] * sbl[i, j, n] + P["Cb_bl"] * bbl[i, j, n] for i, j in IJ))
        for n in N)

    for n in N:
        m += pulp.lpSum(y[i, n] for i in I) <= P["K_tank"], f"R1_{n}"
        for i in I:
            m += u[i, n] <= P["U"], f"R2_{i}_{n}"
            m += P["V_tank"] * (y[i, n] - u[i, n]) == P["V_bot"] * (
                wb[i, n] + pulp.lpSum(wbl[i, j, n] for j in J[i])), f"R3_{i}_{n}"
            m += sb[i, n] == gsb(i, a[n]) + gwb(i, a[n]) - pulp.lpSum(wl[i, j, n] for j in J[i]), f"R4_{i}_{n}"
            m += wb[i, n] <= P["M_b"] * zb[i, n], f"R9_{i}_{n}"
            if n in L:
                m += wb[i, n] == 0, f"R13b_{i}_{n}"
        for i, j in IJ:
            m += (sbl[i, j, n] - bbl[i, j, n] ==
                  gsbl(i, j, a[n]) - gbbl(i, j, a[n]) + gwbl(i, j, a[n]) + wl[i, j, n] - D[i, j, n]), f"R5_{i}_{j}_{n}"
            m += wbl[i, j, n] <= P["M_bl"] * zbl[i, j, n], f"R8_{i}_{j}_{n}"
            m += wl[i, j, n] <= P["M_l"] * zl[i, j, n], f"R10_{i}_{j}_{n}"
            m += zl[i, j, n] <= zpos[i, j], f"R11_{i}_{j}_{n}"
            if n in L:
                m += wbl[i, j, n] == 0, f"R13_{i}_{j}_{n}"
            if contra_pedido:
                m += sbl[i, j, n] == 0, f"R15_{i}_{j}_{n}"
        # R7 horas de linea
        m += (pulp.lpSum(P["Tw_bl"] * wbl[i, j, n] + P["Tz_bl"] * zbl[i, j, n]
                         + P["Tw_l"] * wl[i, j, n] + P["Tz_l"] * zl[i, j, n] for i, j in IJ)
              + pulp.lpSum(P["Tw_b"] * wb[i, n] + P["Tz_b"] * zb[i, n] for i in I)) <= P["T_prod"], f"R7_{n}"
    m += pulp.lpSum(zpos.values()) <= PL, "R12"

    t1 = time.perf_counter()
    m.solve(pulp.HiGHS(msg=verbose, timeLimit=tiempo_max))
    t2 = time.perf_counter()
    status = pulp.LpStatus[m.status]
    variables = m.variables()
    tamano = {"Variables": len(variables),
              "Variables enteras/binarias": sum(v.cat == pulp.LpInteger for v in variables),
              "Restricciones": len(m.constraints),
              "Tiempo construcción (s)": t1 - t0, "Tiempo resolución (s)": t2 - t1}

    # --- Resultados en DataFrames ---
    val = lambda v: (v.value() or 0.0)
    filas = []
    for n in N:
        h = (sum(P["Tw_bl"] * val(wbl[i, j, n]) + P["Tz_bl"] * val(zbl[i, j, n])
                 + P["Tw_l"] * val(wl[i, j, n]) + P["Tz_l"] * val(zl[i, j, n]) for i, j in IJ)
             + sum(P["Tw_b"] * val(wb[i, n]) + P["Tz_b"] * val(zb[i, n]) for i in I))
        hs = (sum(P["Tz_bl"] * val(zbl[i, j, n]) + P["Tz_l"] * val(zl[i, j, n]) for i, j in IJ)
              + sum(P["Tz_b"] * val(zb[i, n]) for i in I))
        filas.append(dict(nodo=n, padre=padre[n], prob=pi[n], horas=h, horas_setup=hs,
                          estanques=sum(val(y[i, n]) for i in I)))
    nodos = pd.DataFrame(filas).set_index("nodo")

    prod = []
    for n in N:
        for i, j in IJ:
            prod.append(dict(nodo=n, vino=i, etiqueta=j, demanda=D[i, j, n],
                             embotella_etiq=val(wbl[i, j, n]), etiqueta_guardadas=val(wl[i, j, n]),
                             inv_terminado=val(sbl[i, j, n]), atraso=val(bbl[i, j, n]),
                             setup_bl=round(val(zbl[i, j, n])), setup_l=round(val(zl[i, j, n]))))
    productos = pd.DataFrame(prod)
    vinos = pd.DataFrame([dict(nodo=n, vino=i, embotella_sin_etiq=val(wb[i, n]), inv_sin_etiq=val(sb[i, n]),
                               setup_b=round(val(zb[i, n])), estanques=val(y[i, n]), vacio_ultimo=val(u[i, n]))
                          for n in N for i in I])

    # descomposicion del costo
    costo = dict(
        setups=sum(pi[n] * (P["Cz_b"] * val(zb[i, n])) for n in N for i in I)
        + sum(pi[n] * (P["Cz_bl"] * val(zbl[i, j, n]) + P["Cz_l"] * val(zl[i, j, n])) for n in N for i, j in IJ),
        inventario=sum(pi[n] * P["Cs_b"] * val(sb[i, n]) for n in N for i in I)
        + sum(pi[n] * P["Cs_bl"] * val(sbl[i, j, n]) for n in N for i, j in IJ),
        atrasos=sum(pi[n] * P["Cb_bl"] * val(bbl[i, j, n]) for n in N for i, j in IJ))

    # los 6 terminos de la funcion objetivo, por separado (valor esperado)
    E = lambda f: sum(pi[n] * f(n) for n in N)
    costo_detalle = {
        "Set-up embotellado sin etiqueta": E(lambda n: sum(P["Cz_b"] * val(zb[i, n]) for i in I)),
        "Set-up embotellado + etiquetado": E(lambda n: sum(P["Cz_bl"] * val(zbl[i, j, n]) for i, j in IJ)),
        "Set-up etiquetado posterior": E(lambda n: sum(P["Cz_l"] * val(zl[i, j, n]) for i, j in IJ)),
        "Inventario sin etiqueta": E(lambda n: sum(P["Cs_b"] * val(sb[i, n]) for i in I)),
        "Inventario terminado": E(lambda n: sum(P["Cs_bl"] * val(sbl[i, j, n]) for i, j in IJ)),
        "Atrasos": E(lambda n: sum(P["Cb_bl"] * val(bbl[i, j, n]) for i, j in IJ)),
    }
    kpi = _calcular_kpis(nodos, productos, vinos, P)

    return dict(status=status, obj=pulp.value(m.objective), costo=costo, costo_detalle=costo_detalle, kpis=kpi,
                tamano=tamano, PL=PL, contra_pedido=contra_pedido,
                nodos=nodos, productos=productos, vinos=vinos,
                postergables=[ij for ij in IJ if val(zpos[ij]) > 0.5],
                padre=padre, hijos=hijos, L=L, IJ=IJ, params=P, modelo=m)


def resumen(res):
    c = res["costo"]
    nom = "Contra pedido" if res["contra_pedido"] else _nombre_politica(res["PL"])
    post = ", ".join(_nombre_producto(i, j) for i, j in res["postergables"]) or "ninguno"
    print(f"Política: {nom}  |  estado: {res['status']}  |  costo esperado: ${_num(res['obj'])}")
    print(f"  Set-ups ${_num(c['setups'])}  |  Inventario ${_num(c['inventario'])}  |  Atrasos ${_num(c['atrasos'])}")
    print(f"  Productos postergables elegidos: {post}")
    k = res["kpis"]
    print(f"  Nivel de servicio {_pct(k['Nivel de servicio'], 1)}  |  Utilización media {_pct(k['Utilización media'])}"
          f"  |  Set-ups esperados {_num(k['Set-ups (total)'], 1)}"
          f"  |  Inventario promedio {_num(k['Inventario promedio (total)'])} botellas")


def comparar_politicas(PLs=(0, 2, 3, 4, 5), **kw):
    """Resuelve a stock, hibridas y contra pedido y devuelve tabla comparativa."""
    filas = []
    corridas = [(_nombre_politica(PL), dict(PL=PL)) for PL in PLs] + [("Contra pedido", dict(contra_pedido=True))]
    for nombre, args in corridas:
        r = resolver(**args, **kw)
        filas.append(dict(politica=nombre, estado=r["status"], costo=r["obj"], **r["costo"],
                          postergables=r["postergables"], **r["costo_detalle"], **r["kpis"], **r["tamano"]))
    return pd.DataFrame(filas).set_index("politica")


COMPONENTES_COSTO = ["Set-up embotellado sin etiqueta", "Set-up embotellado + etiquetado",
                     "Set-up etiquetado posterior", "Inventario sin etiqueta", "Inventario terminado", "Atrasos"]


def _calcular_kpis(nodos, productos, vinos, P):
    """KPIs operativos esperados (ponderados por la probabilidad de cada nodo).
    Como las probabilidades de cada periodo suman 1, sum(pi * x) / n_periodos es el promedio por periodo."""
    pi = nodos["prob"]
    n_per = pi.sum()                                   # = numero de periodos
    pr = productos.assign(pi=productos["nodo"].map(pi))
    vi = vinos.assign(pi=vinos["nodo"].map(pi))
    demanda = (pr["pi"] * pr["demanda"]).sum()
    # demanda del nodo que no se entrega a tiempo (lo atrasado al final del nodo, acotado por la demanda del nodo)
    no_atendida = (pr["pi"] * pr[["demanda", "atraso"]].min(axis=1)).sum()
    atraso_nodo = pr.groupby("nodo")["atraso"].sum()
    inv_b = (vi["pi"] * vi["inv_sin_etiq"]).sum() / n_per
    inv_bl = (pr["pi"] * pr["inv_terminado"]).sum() / n_per
    uso = nodos["horas"] / P["T_prod"]
    return {
        "Nivel de servicio": 1 - no_atendida / demanda,
        "Periodos sin atrasos": (pi * (atraso_nodo < 0.5)).sum() / n_per,
        "Atraso promedio (botellas)": (pr["pi"] * pr["atraso"]).sum() / n_per,
        "Utilización media": (pi * uso).sum() / n_per,
        "Utilización máxima": uso.max(),
        "Horas en set-up": (pi * nodos["horas_setup"]).sum() / max((pi * nodos["horas"]).sum(), 1e-9),
        "Set-ups embotellado sin etiqueta": (vi["pi"] * vi["setup_b"]).sum(),
        "Set-ups embotellado + etiquetado": (pr["pi"] * pr["setup_bl"]).sum(),
        "Set-ups etiquetado posterior": (pr["pi"] * pr["setup_l"]).sum(),
        "Set-ups (total)": (vi["pi"] * vi["setup_b"]).sum() + (pr["pi"] * (pr["setup_bl"] + pr["setup_l"])).sum(),
        "Inventario promedio sin etiqueta": inv_b,
        "Inventario promedio terminado": inv_bl,
        "Inventario promedio (total)": inv_b + inv_bl,
        "Cobertura (periodos de demanda)": (inv_b + inv_bl) / (demanda / n_per),
    }


# ---------------------------------------------------------------------------
# Sensibilidad
# ---------------------------------------------------------------------------
def sensibilidad(parametro="T_prod", valores=(21, 42, 63, 84), PLs=(0, 2, 3, 4, 5), detalle=False, **kw):
    """Costo esperado de cada politica al variar un parametro de la Tabla 2.
    Por defecto: niveles de capacidad de la linea del Anexo A (sec. 6, "Niveles de referencia").
    Devuelve un DataFrame (filas = valor del parametro, columnas = politica) con el costo esperado.
    Con detalle=True devuelve una fila por (valor, politica) con costos, componentes y KPIs."""
    filas = []
    for v in valores:
        for PL in PLs:
            r = resolver(PL=PL, **{parametro: v}, **kw)
            filas.append({parametro: v, "politica": _nombre_politica(PL), "estado": r["status"], "costo": r["obj"],
                          **r["costo_detalle"], **r["kpis"]})
    det = pd.DataFrame(filas).set_index([parametro, "politica"])
    return det if detalle else det["costo"].unstack("politica")[[_nombre_politica(PL) for PL in PLs]]


# ---------------------------------------------------------------------------
# Visualizacion
# ---------------------------------------------------------------------------
# Color principal de la presentacion (vino). Se usa para destacar la politica hibrida y en graficos de una serie.
VINO = "#6d2e46"
# Paleta categorica en tonos de la presentacion (orden fijo, validada para daltonismo en pares adyacentes)
PALETA = ["#8e3d5e", "#d1962a", "#0f9479", "#e58fae", "#3f6fc4"]
# Rampa continua (claro -> oscuro) para el uso de la linea en el arbol
SECUENCIAL = ["#f4e4eb", "#e6c3d1", "#d59db4", "#bf7393", "#a14f71", "#6d2e46", "#4f1f32"]
FONDO_DESTACADO = "#f5e8ee"
SUPERFICIE, TEXTO, TEXTO_2, GRILLA = "#ffffff", "#0b0b0b", "#52514e", "#e6e5e0"

# Cada concepto conserva su color en todos los graficos
COLOR = dict(etiquetada=PALETA[0], sin_etiqueta=PALETA[1], etiquetado_posterior=PALETA[2],
             setups=PALETA[0], inventario=PALETA[1], atrasos=PALETA[2])

DESCRIPCION_PARAMS = dict(
    T_prod="Horas de línea por periodo", K_tank="Estanques disponibles por periodo",
    V_tank="Litros por estanque", V_bot="Litros por botella", U="Fracción máx. de estanque sin usar",
    Tz_b="Set-up embotellado sin etiqueta (h)", Tz_bl="Set-up embotellado + etiquetado (h)",
    Tz_l="Set-up etiquetado posterior (h)", Tw_b="Horas por botella sin etiqueta",
    Tw_bl="Horas por botella embotellada y etiquetada", Tw_l="Horas por botella etiquetada después",
    Cz_b="Costo set-up embotellado sin etiqueta ($)", Cz_bl="Costo set-up embotellado + etiquetado ($)",
    Cz_l="Costo set-up etiquetado posterior ($)", Cs_b="Costo inventario sin etiqueta ($/botella·periodo)",
    Cs_bl="Costo inventario terminado ($/botella·periodo)", Cb_bl="Costo de atraso ($/botella·periodo)",
    M_b="Máx. producción sin etiqueta (botellas)", M_bl="Máx. producción etiquetada (botellas)",
    M_l="Máx. etiquetado posterior (botellas)")


def aplicar_estilo():
    """Estilo sobrio y homogeneo para todos los graficos."""
    plt.rcParams.update({
        "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
        "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 10, "axes.labelsize": 10, "axes.labelcolor": TEXTO_2,
        "text.color": TEXTO, "xtick.color": TEXTO_2, "ytick.color": TEXTO_2,
        "axes.edgecolor": GRILLA, "axes.linewidth": 1, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRILLA, "grid.linewidth": 0.8,
        "grid.linestyle": "-", "axes.axisbelow": True, "axes.prop_cycle": plt.cycler(color=PALETA),
        "legend.frameon": False, "legend.fontsize": 9, "xtick.major.size": 0, "ytick.major.size": 0,
    })


aplicar_estilo()


def _num(x, dec=0):
    """Formato chileno: 1.234.567,8"""
    if round(x, dec) == 0:
        x = 0.0
    return f"{x:,.{dec}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def _pct(x, dec=0):
    return _num(100 * x, dec) + " %"


def _corto(v):
    """Montos compactos para rotulos: 1,9 MM / 13,1 mil / 500"""
    if abs(v) >= 1e6:
        return f"{_num(v / 1e6, 1 if abs(v) < 1e8 else 0)} MM"
    if abs(v) >= 1e3:
        return f"{_num(v / 1e3, 1 if abs(v) < 1e5 else 0)} mil"
    return _num(v)


def _eje_miles(ax, eje="y"):
    fmt = plt.FuncFormatter(lambda x, _: _num(x))
    (ax.yaxis if eje == "y" else ax.xaxis).set_major_formatter(fmt)


def _nombre_producto(i, j):
    return f"Vino {i} · Etiq. {j}"


def _nombre_politica(PL):
    return "A stock" if PL == 0 else f"Híbrida PL={PL}"


def _periodos(padre):
    """Periodo (1, 2, ...) de cada nodo segun su profundidad en el arbol."""
    per = {}
    for n in sorted(padre):
        per[n] = 1 if padre[n] is None else per[padre[n]] + 1
    return per


def _separar_periodos(ax, nodos, padre):
    """Lineas verticales suaves entre periodos y rotulo de cada periodo sobre el eje x."""
    per = _periodos(padre)
    ps = [per[n] for n in nodos]
    for k in range(1, len(ps)):
        if ps[k] != ps[k - 1]:
            ax.axvline(k - 0.5, color=GRILLA, lw=1, zorder=0)
    for p in sorted(set(ps)):
        idx = [k for k, q in enumerate(ps) if q == p]
        ax.text((idx[0] + idx[-1]) / 2, 1.0, f"Periodo {p}", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8, color=TEXTO_2)


def _barras_apiladas(ax, df, colores):
    """Barras apiladas con un fino espacio del color de fondo entre segmentos."""
    x = range(len(df))
    base = pd.Series(0.0, index=df.index)
    for col in df.columns:
        ax.bar(x, df[col], bottom=base, width=0.7, color=colores[col], label=col,
               edgecolor=SUPERFICIE, linewidth=1)
        base += df[col]
    ax.set_xticks(list(x), [f"N{n}" for n in df.index])
    ax.grid(axis="x", visible=False)
    return base


def _posiciones(res):
    """Posiciones para dibujar el arbol (periodos en x, hojas repartidas en y)."""
    hijos, padre = res["hijos"], res["padre"]
    raiz = [n for n in padre if padre[n] is None][0]
    pos, cont = {}, [0]

    def rec(n, d):
        if not hijos[n]:
            pos[n] = (d, -cont[0]); cont[0] += 1
        else:
            for h in hijos[n]:
                rec(h, d + 1)
            pos[n] = (d, sum(pos[h][1] for h in hijos[n]) / len(hijos[n]))
    rec(raiz, 0)
    return pos


def graficar_demanda(datos_mod=datos):
    """Demanda por nodo, apilada por producto."""
    D, J, padre = datos_mod.DEM, datos_mod.ETIQUETAS, datos_mod.PADRE
    N = sorted(padre)
    df = pd.DataFrame({_nombre_producto(i, j): [D[i, j, n] for n in N] for i in J for j in J[i]}, index=N)
    colores = dict(zip(df.columns, PALETA))
    fig, ax = plt.subplots(figsize=(11, 4.2))
    tot = _barras_apiladas(ax, df, colores)
    for k, t in enumerate(tot):
        ax.annotate(_num(t / 1000) + " mil", (k, t), ha="center", va="bottom", fontsize=8, color=TEXTO_2,
                    xytext=(0, 3), textcoords="offset points")
    _separar_periodos(ax, N, padre)
    _eje_miles(ax)
    ax.margins(y=0.12)
    ax.set_title("Demanda por escenario y producto", pad=22)
    ax.set_ylabel("Botellas")
    ax.legend(ncol=len(df.columns), loc="upper center", bbox_to_anchor=(0.5, -0.08))
    plt.tight_layout()
    return fig


def _nombre_res(res):
    return "Contra pedido" if res["contra_pedido"] else _nombre_politica(res["PL"])


def graficar_arbol(res, inventario=True):
    """Arbol de escenarios: color = % de horas de linea usadas en cada nodo.
    Con inventario=True cada nodo muestra tambien las botellas guardadas al final del nodo."""
    from matplotlib.colors import LinearSegmentedColormap
    G = nx.DiGraph([(p, n) for n, p in res["padre"].items() if p is not None])
    pos = _posiciones(res)
    nd, pr = res["nodos"], res["productos"]
    T = res["params"]["T_prod"]
    atraso = pr.groupby("nodo")["atraso"].sum()
    inv = pr.groupby("nodo")["inv_terminado"].sum() + res["vinos"].groupby("nodo")["inv_sin_etiq"].sum()
    uso = {n: nd.loc[n, "horas"] / T for n in G.nodes}
    cmap = LinearSegmentedColormap.from_list("uso", SECUENCIAL)

    tam = 5200 if inventario else 4200
    fig, ax = plt.subplots(figsize=(11, 9.5))
    nx.draw_networkx_edges(G, pos, ax=ax, node_size=tam, arrows=True, arrowstyle="-|>", arrowsize=12,
                           edge_color="#b5b4ae", width=1.2)
    nodos = nx.draw_networkx_nodes(G, pos, ax=ax, node_size=tam, node_color=[uso[n] for n in G.nodes],
                                   cmap=cmap, vmin=0, vmax=1, edgecolors=SUPERFICIE, linewidths=2)
    for n, (x, y) in pos.items():
        claro = uso[n] > 0.45
        txt = (f"Nodo {n}\nπ = {_num(nd.loc[n, 'prob'], 2)}\n{_num(nd.loc[n, 'horas'], 1)} h"
               + (f"\ninv. {_corto(inv[n])}" if inventario else "")
               + (f"\natraso {_corto(atraso[n])}" if atraso[n] > 0.5 else ""))
        ax.text(x, y, txt, ha="center", va="center", fontsize=8, linespacing=1.4,
                color=SUPERFICIE if claro else TEXTO, fontweight="bold" if claro else "normal")
    for p in sorted({x for x, _ in pos.values()}):
        ax.text(p, 1.0, f"Periodo {p + 1}", transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=10, color=TEXTO_2)
    cb = fig.colorbar(nodos, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(f"Uso de la línea (% de {_num(T)} h)")
    cb.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    cb.outline.set_visible(False)
    subt = "probabilidad, horas de línea e inventario al cierre" if inventario else "probabilidad y horas de línea usadas"
    ax.set_title(f"Árbol de escenarios — {_nombre_res(res)}: {subt}", pad=28)
    ax.axis("off")
    ax.margins(x=0.08, y=0.08)
    plt.tight_layout()
    return fig


def graficar_decisiones(res):
    """Produccion, inventario, atrasos y horas de linea por nodo."""
    pr, vi, nd, padre = res["productos"], res["vinos"], res["nodos"], res["padre"]
    nodos = list(nd.index)
    fig, ax = plt.subplots(2, 2, figsize=(14, 8.5))

    # Produccion
    prod = pd.DataFrame({
        "Embotellada y etiquetada": pr.groupby("nodo")["embotella_etiq"].sum(),
        "Embotellada sin etiqueta": vi.groupby("nodo")["embotella_sin_etiq"].sum(),
        "Etiquetada después": pr.groupby("nodo")["etiqueta_guardadas"].sum()}).loc[nodos]
    _barras_apiladas(ax[0, 0], prod, dict(zip(prod.columns, [COLOR["etiquetada"], COLOR["sin_etiqueta"],
                                                              COLOR["etiquetado_posterior"]])))
    ax[0, 0].set_title("Producción por nodo", pad=20); ax[0, 0].set_ylabel("Botellas")

    # Inventario
    inv = pd.DataFrame({
        "Producto terminado": pr.groupby("nodo")["inv_terminado"].sum(),
        "Sin etiqueta": vi.groupby("nodo")["inv_sin_etiq"].sum()}).loc[nodos]
    _barras_apiladas(ax[0, 1], inv, {"Producto terminado": COLOR["etiquetada"],
                                     "Sin etiqueta": COLOR["sin_etiqueta"]})
    ax[0, 1].set_title("Inventario al final de cada nodo", pad=20); ax[0, 1].set_ylabel("Botellas")

    # Atrasos
    atr = pr.pivot_table(index="nodo", columns=["vino", "etiqueta"], values="atraso").loc[nodos]
    atr.columns = [_nombre_producto(i, j) for i, j in atr.columns]
    a = ax[1, 0]
    if atr.values.sum() < 0.5:
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
        for s in a.spines.values():
            s.set_visible(False)
        a.text(0.5, 0.5, "Sin atrasos\nToda la demanda se cumple a tiempo\nen todos los escenarios",
               ha="center", va="center", transform=a.transAxes, fontsize=12, color=TEXTO_2, linespacing=1.6)
        a.set_title("Atrasos por producto")
    else:
        _barras_apiladas(a, atr, dict(zip(atr.columns, PALETA)))
        a.set_title("Atrasos por producto", pad=20); a.set_ylabel("Botellas atrasadas")

    # Horas de linea
    a = ax[1, 1]
    T = res["params"]["T_prod"]
    a.bar(range(len(nodos)), nd["horas"], width=0.7, color=VINO)
    a.axhline(T, color=TEXTO_2, lw=1.2)
    a.annotate(f"Capacidad: {_num(T)} h", (len(nodos) - 0.6, T), ha="right", va="bottom", fontsize=9,
               color=TEXTO_2, xytext=(0, 3), textcoords="offset points")
    k = int(nd["horas"].values.argmax())
    hk = nd["horas"].iloc[k]
    # si la barra llega cerca de la capacidad, el rotulo va sobre la linea de capacidad para no taparla
    a.annotate(f"{_num(hk, 1)} h ({hk / T:.0%})", (k, max(hk, T) if hk > 0.85 * T else hk), ha="center",
               va="bottom", fontsize=9, fontweight="bold", xytext=(0, 3), textcoords="offset points")
    a.set_xticks(range(len(nodos)), [f"N{n}" for n in nodos]); a.grid(axis="x", visible=False)
    a.set_ylim(0, T * 1.15)
    a.set_title("Horas de línea usadas", pad=20); a.set_ylabel("Horas")

    for a in ax.flat:
        if a.get_xticks().size:
            _separar_periodos(a, nodos, padre)
            if a is not ax[1, 1]:
                _eje_miles(a)
                a.legend(loc="upper right")
    fig.suptitle(f"Decisiones óptimas por nodo — {_nombre_politica(res['PL']) if not res['contra_pedido'] else 'Contra pedido'}",
                 x=0.01, ha="left", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def tabla_plan_inicial(res):
    """Decision 'aqui y ahora' (nodo raiz): que embotellar y etiquetar en el periodo 1."""
    raiz = [n for n in res["padre"] if res["padre"][n] is None][0]
    pr = res["productos"].query("nodo == @raiz")
    vi = res["vinos"].query("nodo == @raiz").set_index("vino")
    filas = []
    for _, r in pr.iterrows():
        filas.append({"Producto": _nombre_producto(int(r.vino), int(r.etiqueta)), "Demanda": r.demanda,
                      "Embotellar y etiquetar": r.embotella_etiq,
                      "Postergable": "Sí" if (int(r.vino), int(r.etiqueta)) in res["postergables"] else "No"})
    for i, r in vi.iterrows():
        filas.append({"Producto": f"Vino {i} · sin etiqueta", "Demanda": float("nan"),
                      "Embotellar y etiquetar": r.embotella_sin_etiq, "Postergable": ""})
    df = pd.DataFrame(filas).set_index("Producto").rename(columns={"Embotellar y etiquetar": "Botellas a producir"})
    return df.style.format({"Demanda": lambda v: "" if pd.isna(v) else _num(v),
                            "Botellas a producir": _num})


def tabla_comparacion(tabla):
    """Tabla de comparacion de politicas con formato para presentar."""
    base = tabla.loc["A stock", "costo"] if "A stock" in tabla.index else tabla["costo"].iloc[0]
    df = pd.DataFrame({
        "Costo esperado ($)": tabla["costo"], "Set-ups ($)": tabla["setups"],
        "Inventario ($)": tabla["inventario"], "Atrasos ($)": tabla["atrasos"],
        "vs. A stock": tabla["costo"] / base - 1,
        "Productos postergables": [", ".join(f"V{i}E{j}" for i, j in p) or "—" for p in tabla["postergables"]]})
    mejor = df["Costo esperado ($)"].idxmin()
    fmt = {c: _num for c in df.columns[:4]}
    fmt["vs. A stock"] = lambda v: "—" if abs(v) < 1e-9 else ("+" if v > 0 else "−") + _num(abs(v) * 100) + " %"
    return (df.rename_axis("Política").style
            .format(fmt)
            .apply(lambda f: [f"font-weight: bold; background-color: {FONDO_DESTACADO}" if f.name == mejor else ""] * len(f),
                   axis=1))


def tabla_parametros(params):
    return pd.DataFrame({"Descripción": {k: DESCRIPCION_PARAMS.get(k, "") for k in params},
                         "Valor": {k: (_num(v, 5).rstrip("0").rstrip(",") if v != int(v) else _num(v))
                                   for k, v in params.items()}}).rename_axis("Parámetro")


def graficar_comparacion(tabla, factor_outlier=20):
    """Costo esperado por politica.
    Izquierda: desglose apilado (MM$) de las politicas comparables; las que cuestan mas de
    `factor_outlier` veces la mas barata se omiten porque aplastarian al resto.
    Derecha: costo total de todas las politicas en escala logaritmica."""
    normales = tabla[tabla["costo"] <= factor_outlier * tabla["costo"].min()]
    omitidas = [p for p in tabla.index if p not in normales.index]
    base = tabla["costo"].iloc[0]

    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw=dict(width_ratios=[3, 2]))
    comp = {"Set-ups": "setups", "Inventario": "inventario", "Atrasos": "atrasos"}
    df = pd.DataFrame({k: normales[v] / 1e6 for k, v in comp.items()})
    x = range(len(df))
    abajo = pd.Series(0.0, index=df.index)
    for k, col in comp.items():
        if df[k].sum() > 0:
            ax[0].bar(x, df[k], bottom=abajo, width=0.6, color=COLOR[col], label=k,
                      edgecolor=SUPERFICIE, linewidth=1)
            abajo += df[k]
    for k, (p, t) in enumerate(abajo.items()):
        etiqueta = f"${_num(t, 2)} MM"
        if k > 0:
            etiqueta += f"\n{normales.loc[p, 'costo'] / base - 1:+.0%}".replace(".", ",")
        ax[0].annotate(etiqueta, (k, t), ha="center", va="bottom", fontsize=9, xytext=(0, 3),
                       textcoords="offset points", fontweight="bold" if p == tabla["costo"].idxmin() else "normal")
    ax[0].set_xticks(list(x), df.index); ax[0].grid(axis="x", visible=False)
    ax[0].margins(y=0.18)
    ax[0].set_ylabel("Costo esperado (millones de $)")
    ax[0].legend(loc="upper right")
    titulo = "Desglose del costo esperado"
    if omitidas:
        titulo += f"  (sin {', '.join(o.lower() for o in omitidas)})"
    ax[0].set_title(titulo)

    a = ax[1]
    colores = ["#c9c8c2" if p in omitidas else VINO for p in tabla.index]
    a.bar(range(len(tabla)), tabla["costo"], width=0.6, color=colores)
    a.set_yscale("log")
    a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${_num(v / 1e6, 0 if v >= 1e6 else 1)} MM"))
    for k, t in enumerate(tabla["costo"]):
        a.annotate(_num(t / 1e6, 0 if t >= 1e8 else 2), (k, t), ha="center", va="bottom", fontsize=8.5,
                   xytext=(0, 3), textcoords="offset points")
    a.set_xticks(range(len(tabla)), tabla.index, rotation=25, ha="right"); a.grid(axis="x", visible=False)
    a.margins(y=0.15)
    a.set_title("Costo total, todas las políticas (escala log)")
    fig.suptitle("Comparación de políticas", x=0.01, ha="left", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def graficar_sensibilidad(tabla, base=None):
    """Izquierda: costo esperado de cada politica por nivel del parametro (escala log).
    Derecha: ahorro de cada politica hibrida frente a 'A stock' (beneficio de postergar)."""
    import numpy as np
    param = tabla.index.name
    niveles, pols = list(tabla.index), list(tabla.columns)
    x = np.arange(len(niveles))
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))

    def barras(a, df, cols):
        w = 0.8 / len(cols)
        for k, col in enumerate(cols):
            a.bar(x + (k - (len(cols) - 1) / 2) * w, df[col], width=w, label=col,
                  color=_color_pol(col), edgecolor=SUPERFICIE, linewidth=1)
        etiquetas = [f"{_num(v)} h" + ("\n(caso base)" if v == base else "") for v in niveles]
        a.set_xticks(x, etiquetas); a.grid(axis="x", visible=False)
        a.set_xlabel(DESCRIPCION_PARAMS.get(param, param) + f" ({param})")

    a = ax[0]
    barras(a, tabla, pols)
    a.set_yscale("log")
    a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${_num(v / 1e6, 0 if v >= 1e6 else 1)} MM"))
    a.set_ylabel("Costo esperado (escala log)")
    a.set_title("Costo esperado por política")

    a = ax[1]
    hibridas = [c for c in pols if c != "A stock"]
    ahorro = (1 - tabla[hibridas].div(tabla["A stock"], axis=0)) * 100
    barras(a, ahorro, hibridas)
    for k, nivel in enumerate(niveles):
        col = ahorro.loc[nivel].idxmax()
        dx = (hibridas.index(col) - (len(hibridas) - 1) / 2) * 0.8 / len(hibridas)
        a.annotate(f"{_num(ahorro.loc[nivel, col])} %", (k + dx, ahorro.loc[nivel, col]), ha="center",
                   va="bottom", fontsize=9, fontweight="bold", xytext=(0, 3), textcoords="offset points")
    a.set_ylim(0, 110); a.set_yticks(range(0, 101, 20))
    a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f} %"))
    a.set_ylabel("Ahorro vs. A stock")
    a.set_title("Beneficio de postergar: ahorro frente a A stock")

    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.01, 0.95), ncol=len(pols))
    titulo = "Sensibilidad a la capacidad de la línea" if param == "T_prod" else f"Sensibilidad a {param}"
    fig.suptitle(titulo, x=0.01, ha="left",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


# ---------------------------------------------------------------------------
# Desglose de costos y KPIs operativos
# ---------------------------------------------------------------------------
GRUPOS_KPI = {
    "Servicio": ["Nivel de servicio", "Periodos sin atrasos", "Atraso promedio (botellas)"],
    "Utilización de la línea": ["Utilización media", "Utilización máxima", "Horas en set-up"],
    "Set-ups esperados": ["Set-ups embotellado sin etiqueta", "Set-ups embotellado + etiquetado",
                          "Set-ups etiquetado posterior", "Set-ups (total)"],
    "Inventario promedio por periodo (botellas)": ["Inventario promedio sin etiqueta", "Inventario promedio terminado",
                                                   "Inventario promedio (total)", "Cobertura (periodos de demanda)"],
}
_FMT_KPI = {**{k: _pct for k in ["Nivel de servicio", "Periodos sin atrasos", "Utilización media",
                                 "Utilización máxima", "Horas en set-up"]},
            **{k: (lambda v: _num(v, 1)) for k in GRUPOS_KPI["Set-ups esperados"]},
            "Cobertura (periodos de demanda)": lambda v: _num(v, 2)}
_DESTACADO = f"font-weight: bold; background-color: {FONDO_DESTACADO}"


def tabla_costos(tabla):
    """Los 6 terminos de la funcion objetivo por politica (valor esperado, $). Filas = componentes."""
    df = tabla[COMPONENTES_COSTO].T
    df.loc["Costo total"] = tabla["costo"]
    df.columns.name = None
    mejor = tabla["costo"].idxmin()
    return (df.rename_axis("Componente ($)").style
            .format(_num)
            .apply(lambda c: [_DESTACADO if c.name == mejor else ""] * len(c))
            .apply(lambda f: ["border-top: 2px solid #52514e; font-weight: bold" if f.name == "Costo total" else ""]
                   * len(f), axis=1))


def tabla_kpis(tabla):
    """KPIs operativos por politica, agrupados por tema. Filas = KPI, columnas = politica."""
    filas = [(g, k) for g, ks in GRUPOS_KPI.items() for k in ks]
    df = pd.DataFrame({p: [tabla.loc[p, k] for _, k in filas] for p in tabla.index},
                      index=pd.MultiIndex.from_tuples(filas, names=["", "KPI"]))
    texto = df.copy().astype(object)
    for g, k in filas:
        texto.loc[(g, k)] = [_FMT_KPI.get(k, _num)(v) for v in df.loc[(g, k)]]
    mejor = tabla["costo"].idxmin()
    return texto.style.apply(lambda c: [_DESTACADO if c.name == mejor else ""] * len(c))


def _barras_politicas(a, serie, fmt, omitidas=()):
    """Barras de una sola serie por politica, con rotulo del valor."""
    colores = ["#c9c8c2" if p in omitidas else VINO for p in serie.index]
    a.bar(range(len(serie)), serie.values, width=0.6, color=colores)
    for k, v in enumerate(serie.values):
        a.annotate(fmt(v), (k, v), ha="center", va="bottom", fontsize=8.5, xytext=(0, 2), textcoords="offset points")
    a.set_xticks(range(len(serie)), serie.index, rotation=25, ha="right")
    a.grid(axis="x", visible=False)
    a.margins(y=0.18)


def graficar_costos_detalle(tabla):
    """Un panel por cada termino de la funcion objetivo (escala propia en cada uno)."""
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5), sharex=True)
    for a, comp in zip(ax.flat, COMPONENTES_COSTO):
        _barras_politicas(a, tabla[comp], lambda v: "$" + _corto(v), omitidas=["Contra pedido"])
        a.set_title(comp, fontsize=11)
        a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: _corto(v)))
    for a in ax[0]:
        a.tick_params(labelbottom=False)
    fig.suptitle("Costo esperado por componente de la función objetivo ($)", x=0.01, ha="left",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def _pct_eje(a):
    a.set_ylim(0, 112)
    a.set_yticks(range(0, 101, 20))
    a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f} %"))


def _apiladas_politicas(a, tabla, series, fmt):
    pols = list(tabla.index)
    x = range(len(pols))
    abajo = pd.Series(0.0, index=pols)
    for nombre, (col, c) in series.items():
        a.bar(x, tabla[col], bottom=abajo, width=0.6, color=c, label=nombre, edgecolor=SUPERFICIE, linewidth=1)
        abajo += tabla[col]
    for k, v in enumerate(abajo):
        a.annotate(fmt(v), (k, v), ha="center", va="bottom", fontsize=8.5, xytext=(0, 2), textcoords="offset points")
    a.set_xticks(list(x), pols, rotation=25, ha="right")
    a.grid(axis="x", visible=False)


def graficar_kpis(tabla):
    """KPIs operativos por politica: servicio, utilizacion, set-ups e inventario promedio."""
    fig, ax = plt.subplots(2, 2, figsize=(15, 9.5))
    pols = list(tabla.index)
    x = range(len(pols))

    a = ax[0, 0]
    _barras_politicas(a, tabla["Nivel de servicio"] * 100, lambda v: _num(v, 1) + " %", omitidas=["Contra pedido"])
    _pct_eje(a)
    a.set_title("Nivel de servicio (demanda entregada a tiempo)")

    a = ax[0, 1]
    w = 0.38
    a.bar([k - w / 2 for k in x], tabla["Utilización media"] * 100, width=w, color=PALETA[0], label="Media",
          edgecolor=SUPERFICIE, linewidth=1)
    a.bar([k + w / 2 for k in x], tabla["Utilización máxima"] * 100, width=w, color=PALETA[1],
          label="Máxima (nodo más cargado)", edgecolor=SUPERFICIE, linewidth=1)
    _pct_eje(a)
    a.set_xticks(list(x), pols, rotation=25, ha="right")
    a.grid(axis="x", visible=False)
    a.legend(loc="upper left", ncol=2)
    a.set_title("Utilización de la línea (% de las horas disponibles)")

    a = ax[1, 0]
    _apiladas_politicas(a, tabla, {
        "Embotellado + etiquetado": ("Set-ups embotellado + etiquetado", COLOR["etiquetada"]),
        "Embotellado sin etiqueta": ("Set-ups embotellado sin etiqueta", COLOR["sin_etiqueta"]),
        "Etiquetado posterior": ("Set-ups etiquetado posterior", COLOR["etiquetado_posterior"])},
        lambda v: _num(v, 1))
    a.margins(y=0.25)
    a.legend(loc="upper left", ncol=3)
    a.set_title("Número esperado de set-ups en el horizonte")

    a = ax[1, 1]
    _apiladas_politicas(a, tabla, {
        "Producto terminado": ("Inventario promedio terminado", COLOR["etiquetada"]),
        "Sin etiqueta": ("Inventario promedio sin etiqueta", COLOR["sin_etiqueta"])}, _num)
    _eje_miles(a)
    a.set_ylim(0, tabla["Inventario promedio (total)"].max() * 1.15)
    a.legend(loc="upper right")
    a.set_title("Inventario promedio por periodo (botellas)")

    fig.suptitle("KPIs operativos por política (valores esperados)", x=0.01, ha="left", fontsize=14,
                 fontweight="bold")
    plt.tight_layout()
    return fig


def graficar_sensibilidad_kpis(det, base=None):
    """Nivel de servicio, inventario promedio y utilizacion media por nivel del parametro y politica."""
    import numpy as np
    param = det.index.names[0]
    niveles = list(det.index.get_level_values(0).unique())
    pols = list(det.index.get_level_values(1).unique())
    x = np.arange(len(niveles))
    w = 0.8 / len(pols)
    paneles = [("Nivel de servicio", "Nivel de servicio", True),
               ("Inventario promedio (total)", "Inventario promedio por periodo (botellas)", False),
               ("Utilización media", "Utilización media de la línea", True)]
    fig, ax = plt.subplots(1, 3, figsize=(16, 5.2))
    for a, (col, titulo, es_pct) in zip(ax, paneles):
        tab = det[col].unstack("politica")[pols] * (100 if es_pct else 1)
        for k, p in enumerate(pols):
            a.bar(x + (k - (len(pols) - 1) / 2) * w, tab.loc[niveles, p], width=w, color=_color_pol(p), label=p,
                  edgecolor=SUPERFICIE, linewidth=1)
        a.set_xticks(x, [f"{_num(v)} h" + ("\n(caso base)" if v == base else "") for v in niveles])
        a.grid(axis="x", visible=False)
        a.set_xlabel(DESCRIPCION_PARAMS.get(param, param))
        if es_pct:
            _pct_eje(a)
        else:
            _eje_miles(a)
        a.set_title(titulo, fontsize=11)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.01, 0.95), ncol=len(pols))
    fig.suptitle("KPIs operativos según la capacidad de la línea" if param == "T_prod" else f"KPIs según {param}",
                 x=0.01, ha="left", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


# ---------------------------------------------------------------------------
# Graficos para la presentacion: caso base (A stock) vs. hibrida
# ---------------------------------------------------------------------------
# Cada politica tiene siempre el mismo color: grises para contra pedido y a stock, y una rampa de vino
# (validada como rampa ordinal) para las hibridas, mas oscura mientras mas etiquetas se postergan.
COLOR_POLITICA = {"Contra pedido": "#c9c8c2", "A stock": "#8f8e88", "Híbrida PL=2": "#d38aa6",
                  "Híbrida PL=3": "#b45e80", "Híbrida PL=4": "#8e3d5e", "Híbrida PL=5": VINO}


AHORRO, COSTO_EXTRA = PALETA[2], PALETA[1]


def _color_pol(p):
    return COLOR_POLITICA.get(p, VINO)


def graficar_politicas(tabla, politicas=("Contra pedido", "A stock", "Híbrida PL=5")):
    """Costo esperado de las tres politicas (escala log) con el desglose escrito bajo cada barra."""
    t = tabla.loc[list(politicas)]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    x = range(len(t))
    ax.bar(x, t["costo"], width=0.55, color=[_color_pol(p) for p in t.index])
    ax.set_yscale("log")
    ax.set_ylim(t["costo"].min() / 3, t["costo"].max() * 6)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"${_num(v / 1e6, 0 if v >= 1e6 else 1)} MM"))
    ref = t.loc["A stock", "costo"] if "A stock" in t.index else None
    for k, (pol, fila) in enumerate(t.iterrows()):
        txt = f"${_corto(fila['costo'])}"
        if ref and pol != "A stock":
            r = fila["costo"] / ref
            txt += f"\n{_num(r)}× el caso base" if r >= 2 else f"\n{_pct(r - 1)} vs. caso base".replace("-", "−")
        elif pol == "A stock":
            txt += "\n(caso base)"
        ax.annotate(txt, (k, fila["costo"]), ha="center", va="bottom", fontsize=11, fontweight="bold",
                    xytext=(0, 4), textcoords="offset points", linespacing=1.4)
        partes = [("atrasos", fila["Atrasos"]), ("inventario", fila["inventario"]), ("set-ups", fila["setups"])]
        mayor = max(partes, key=lambda z: z[1])
        ax.annotate(f"{_pct(mayor[1] / fila['costo'])} {mayor[0]}", (k, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -38), textcoords="offset points", ha="center", va="top", fontsize=9, color=TEXTO_2)
    ax.set_xticks(list(x), t.index, fontsize=11)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Costo esperado (escala log)")
    ax.set_title("Costo esperado de cada política")
    plt.tight_layout()
    return fig


_KPIS_PRES = [  # (columna, titulo, formato, menor es mejor)
    ("costo", "Costo esperado", lambda v: "$" + _corto(v), True),
    ("Nivel de servicio", "Nivel de servicio", lambda v: _pct(v, 1), False),
    ("Inventario promedio (total)", "Inventario promedio (botellas)", _num, True),
    ("Cobertura (periodos de demanda)", "Cobertura (periodos de demanda)", lambda v: _num(v, 2), True),
    ("Set-ups (total)", "Set-ups esperados", lambda v: _num(v, 1), True),
    ("Utilización media", "Utilización media de la línea", _pct, None),
    ("Utilización máxima", "Utilización máxima (nodo más cargado)", _pct, None),
    ("Horas en set-up", "Horas de línea en set-up", _pct, True),
]


_KPIS_PCT = {"Nivel de servicio", "Utilización media", "Utilización máxima", "Horas en set-up"}


def _variacion(b, a, col=None):
    """Variacion de b a a: en puntos porcentuales si el KPI ya es un %, si no en % relativo."""
    if abs(a - b) < 1e-9 or (col is None and abs(a - b) < 0.5):
        return "sin cambio"
    if col in _KPIS_PCT:
        d = _num(100 * (a - b))
        return ("+" if a > b else "") + d.replace("-", "−") + " pp"
    if not b:
        return "nuevo"
    return ("+" if a > b else "") + _pct(a / b - 1).replace("-", "−")


def tabla_base_vs_alternativa(tabla, base="A stock", alt="Híbrida PL=5"):
    """KPIs y costo del caso base vs. la alternativa, con la variacion."""
    filas = []
    for col, nombre, fmt, _ in _KPIS_PRES:
        b, a = tabla.loc[base, col], tabla.loc[alt, col]
        filas.append({"KPI": nombre, base: fmt(b), alt: fmt(a), "Variación": _variacion(b, a, col)})
    for col in COMPONENTES_COSTO:
        b, a = tabla.loc[base, col], tabla.loc[alt, col]
        filas.append({"KPI": f"Costo: {col.lower()}", base: "$" + _num(b), alt: "$" + _num(a),
                      "Variación": _variacion(b, a)})
    df = pd.DataFrame(filas).set_index("KPI")
    return df.style.set_properties(subset=[alt], **{"font-weight": "bold", "background-color": FONDO_DESTACADO})


def graficar_base_vs_alternativa(tabla, base="A stock", alt="Híbrida PL=5"):
    """Un panel por KPI: caso base (gris) vs. alternativa (azul), con la variacion en el titulo."""
    fig, ax = plt.subplots(2, 4, figsize=(16, 7.8))
    for a, (col, titulo, fmt, menor_mejor) in zip(ax.flat, _KPIS_PRES):
        vals = [tabla.loc[base, col], tabla.loc[alt, col]]
        a.bar([0, 1], vals, width=0.6, color=[_color_pol(base), _color_pol(alt)])
        for k, v in enumerate(vals):
            a.annotate(fmt(v), (k, v), ha="center", va="bottom", fontsize=10, fontweight="bold" if k else "normal",
                       xytext=(0, 3), textcoords="offset points")
        a.set_xticks([0, 1], [base, alt])
        a.set_yticks([])
        a.grid(False)
        a.spines["left"].set_visible(False)
        a.set_ylim(0, max(vals) * 1.3 if max(vals) > 0 else 1)
        cambio = _variacion(vals[0], vals[1], col)
        a.set_title(f"{titulo}\n{cambio}", fontsize=10.5, linespacing=1.5)
    fig.suptitle(f"Caso base ({base}) vs. {alt}", x=0.01, ha="left", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def graficar_puente_costos(tabla, base="A stock", alt="Híbrida PL=5"):
    """Grafico de cascada: como se pasa del costo del caso base al de la alternativa, componente a componente."""
    pasos = [(c, tabla.loc[alt, c] - tabla.loc[base, c]) for c in COMPONENTES_COSTO]
    pasos = [(c, d) for c, d in pasos if abs(d) >= 1]
    etiquetas = [base] + [c for c, _ in pasos] + [alt]
    fig, ax = plt.subplots(figsize=(13, 5.8))
    nivel = tabla.loc[base, "costo"]
    ax.bar(0, nivel, width=0.6, color=_color_pol(base))
    ax.annotate("$" + _corto(nivel), (0, nivel), ha="center", va="bottom", xytext=(0, 3), textcoords="offset points",
                fontsize=10)
    for k, (c, d) in enumerate(pasos, start=1):
        color = AHORRO if d < 0 else COSTO_EXTRA
        ax.bar(k, abs(d), bottom=min(nivel, nivel + d), width=0.6, color=color)
        ax.plot([k - 1 + 0.3, k - 0.3], [nivel, nivel], color=TEXTO_2, lw=0.8)
        signo = "−" if d < 0 else "+"
        ax.annotate(f"{signo}${_corto(abs(d))}", (k, max(nivel, nivel + d)), ha="center", va="bottom",
                    xytext=(0, 3), textcoords="offset points", fontsize=10, fontweight="bold")
        nivel += d
    k = len(pasos) + 1
    ax.plot([k - 1 + 0.3, k - 0.3], [nivel, nivel], color=TEXTO_2, lw=0.8)
    ax.bar(k, nivel, width=0.6, color=_color_pol(alt))
    ahorro = 1 - nivel / tabla.loc[base, "costo"]
    ax.annotate(f"${_corto(nivel)}\n(−{_pct(ahorro)})", (k, nivel), ha="center", va="bottom", xytext=(0, 3),
                textcoords="offset points", fontsize=10, fontweight="bold")
    etiquetas = [e.replace(" + ", " +\n").replace("Set-up ", "Set-up\n").replace("Inventario ", "Inventario\n")
                 for e in etiquetas]
    ax.set_xticks(range(len(etiquetas)), etiquetas, fontsize=9)
    ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "$" + _corto(v)))
    ax.margins(y=0.12)
    ax.set_ylabel("Costo esperado")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=AHORRO, label="Ahorro"), Patch(color=COSTO_EXTRA, label="Costo adicional")],
              loc="upper right")
    ax.set_title(f"De dónde viene el ahorro: {base} → {alt}")
    plt.tight_layout()
    return fig


def graficar_inventario_nodos(res_base, res_alt):
    """Inventario al cierre de cada nodo, caso base vs. alternativa, separado en terminado y sin etiqueta."""
    import numpy as np
    nombres = [_nombre_res(res_base), _nombre_res(res_alt)]
    nodos = list(res_base["nodos"].index)
    x = np.arange(len(nodos))
    w = 0.38
    fig, ax = plt.subplots(figsize=(13, 5.5))
    tope = 0
    for k, (res, nombre) in enumerate(zip((res_base, res_alt), nombres)):
        term = res["productos"].groupby("nodo")["inv_terminado"].sum().loc[nodos]
        sin = res["vinos"].groupby("nodo")["inv_sin_etiq"].sum().loc[nodos]
        dx = (k - 0.5) * w
        c = _color_pol(nombre)
        ax.bar(x + dx, term, width=w, color=c, edgecolor=SUPERFICIE, linewidth=1,
               label=f"{nombre}: terminado")
        ax.bar(x + dx, sin, bottom=term, width=w, color=c, edgecolor=SUPERFICIE, linewidth=1,
               hatch="//", label=f"{nombre}: sin etiqueta")
        tope = max(tope, (term + sin).max())
    ax.set_xticks(x, [f"N{n}" for n in nodos])
    ax.grid(axis="x", visible=False)
    _separar_periodos(ax, nodos, res_base["padre"])
    _eje_miles(ax)
    ax.set_ylim(0, tope * 1.1)
    ax.set_ylabel("Botellas guardadas al cierre del nodo")
    handles, labels = ax.get_legend_handles_labels()
    usados = [(h, l) for h, l in zip(handles, labels)
              if not (l.endswith("sin etiqueta") and l.startswith(nombres[0]) and
                      res_base["vinos"]["inv_sin_etiq"].sum() < 0.5)
              and not (l.endswith("terminado") and l.startswith(nombres[1]) and
                       res_alt["productos"]["inv_terminado"].sum() < 0.5)]
    ax.legend(*zip(*usados), loc="upper right")
    ax.set_title("Inventario al cierre de cada escenario", pad=22)
    plt.tight_layout()
    return fig


def tabla_tiempos(tabla):
    """Tamano del modelo y tiempos de resolucion por politica."""
    cols = ["Variables", "Variables enteras/binarias", "Restricciones", "Tiempo construcción (s)",
            "Tiempo resolución (s)"]
    df = tabla[cols + ["estado"]].rename(columns={"estado": "Estado"}).rename_axis("Política")
    return df.style.format({c: _num for c in cols[:3]} | {c: (lambda v: _num(v, 2)) for c in cols[3:]})


def info_equipo():
    """Procesador, nucleos y RAM del equipo, y versiones de Python y del solver."""
    import os, platform, sys, importlib.metadata as md
    cpu = platform.processor()
    ram = None
    try:
        if os.name == "nt":
            import subprocess
            ps = ("$c=Get-CimInstance Win32_Processor; $s=Get-CimInstance Win32_ComputerSystem; "
                  "$c.Name.Trim(); $c.NumberOfCores; $c.NumberOfLogicalProcessors; $s.TotalPhysicalMemory")
            out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                                 timeout=20).stdout.split("\n")
            out = [o.strip() for o in out if o.strip()]
            cpu = f"{out[0]} ({out[1]} núcleos / {out[2]} hilos)"
            ram = int(out[3]) / 1024 ** 3
    except Exception:
        pass
    filas = {"Procesador": cpu, "Memoria RAM": f"{_num(ram, 1)} GB" if ram else "—",
             "Sistema operativo": f"{platform.system()} {platform.release()}",
             "Python": platform.python_version(),
             "PuLP / HiGHS": f"{md.version('pulp')} / {md.version('highspy')}"}
    return pd.Series(filas, name="Equipo").to_frame()
