"""
Modelo de embotellado/etiquetado con postergacion (formulacion_compacta.pdf).
Se resuelve con PuLP + HiGHS (gratis, sin Gurobi).

Uso rapido:
    import modelo
    res = modelo.resolver(PL=2)
    modelo.graficar_arbol(res); modelo.graficar_decisiones(res)
"""
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

    m.solve(pulp.HiGHS(msg=verbose, timeLimit=tiempo_max))
    status = pulp.LpStatus[m.status]

    # --- Resultados en DataFrames ---
    val = lambda v: (v.value() or 0.0)
    filas = []
    for n in N:
        h = (sum(P["Tw_bl"] * val(wbl[i, j, n]) + P["Tz_bl"] * val(zbl[i, j, n])
                 + P["Tw_l"] * val(wl[i, j, n]) + P["Tz_l"] * val(zl[i, j, n]) for i, j in IJ)
             + sum(P["Tw_b"] * val(wb[i, n]) + P["Tz_b"] * val(zb[i, n]) for i in I))
        filas.append(dict(nodo=n, padre=padre[n], prob=pi[n], horas=h,
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

    return dict(status=status, obj=pulp.value(m.objective), costo=costo, PL=PL, contra_pedido=contra_pedido,
                nodos=nodos, productos=productos, vinos=vinos,
                postergables=[ij for ij in IJ if val(zpos[ij]) > 0.5],
                padre=padre, hijos=hijos, L=L, IJ=IJ, params=P, modelo=m)


def resumen(res):
    c = res["costo"]
    nom = "contra pedido" if res["contra_pedido"] else ("a stock" if res["PL"] == 0 else f"hibrida PL={res['PL']}")
    print(f"Politica: {nom} | estado: {res['status']} | costo esperado: ${res['obj']:,.0f}")
    print(f"  set-ups ${c['setups']:,.0f} | inventario ${c['inventario']:,.0f} | atrasos ${c['atrasos']:,.0f}")
    print(f"  productos postergables elegidos (vino, etiqueta): {res['postergables']}")


def comparar_politicas(PLs=(0, 2, 3, 4, 5), **kw):
    """Resuelve a stock, hibridas y contra pedido y devuelve tabla comparativa."""
    filas = []
    for PL in PLs:
        r = resolver(PL=PL, **kw)
        filas.append(dict(politica="a stock" if PL == 0 else f"hibrida PL={PL}", estado=r["status"], costo=r["obj"],
                          **r["costo"], postergables=r["postergables"]))
    r = resolver(contra_pedido=True, **kw)
    filas.append(dict(politica="contra pedido", estado=r["status"], costo=r["obj"], **r["costo"], postergables=[]))
    return pd.DataFrame(filas).set_index("politica")


# ---------------------------------------------------------------------------
# Visualizacion
# ---------------------------------------------------------------------------
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


def graficar_arbol(res):
    """Arbol de escenarios con probabilidad, horas usadas y atraso por nodo."""
    G = nx.DiGraph([(p, n) for n, p in res["padre"].items() if p is not None])
    pos = _posiciones(res)
    nd, pr = res["nodos"], res["productos"]
    atraso = pr.groupby("nodo")["atraso"].sum()
    etiquetas = {n: f"Nodo {n}\n$\\pi$={nd.loc[n,'prob']:.3f}\n{nd.loc[n,'horas']:.1f} h"
                    + (f"\natraso {atraso[n]:,.0f}" if atraso[n] > 0.5 else "") for n in G.nodes}
    colores = [nd.loc[n, "horas"] / res["params"]["T_prod"] for n in G.nodes]
    fig, ax = plt.subplots(figsize=(11, 8))
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, node_size=4500, arrowstyle="-|>")
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=4500, node_color=colores, cmap="YlOrRd", vmin=0, vmax=1,
                           edgecolors="k")
    nx.draw_networkx_labels(G, pos, etiquetas, ax=ax, font_size=8)
    ax.set_title("Arbol de escenarios (color = % de horas de linea usadas)")
    ax.axis("off")
    plt.tight_layout()
    return fig


def graficar_decisiones(res):
    """Produccion, inventarios, atrasos y horas por nodo."""
    pr, vi, nd = res["productos"], res["vinos"], res["nodos"]
    nodos = list(nd.index)
    fig, ax = plt.subplots(2, 2, figsize=(14, 8))

    prod = pr.groupby("nodo")[["embotella_etiq", "etiqueta_guardadas"]].sum()
    prod["embotella_sin_etiq"] = vi.groupby("nodo")["embotella_sin_etiq"].sum()
    prod[["embotella_etiq", "embotella_sin_etiq", "etiqueta_guardadas"]].plot.bar(ax=ax[0, 0], stacked=True)
    ax[0, 0].set_title("Produccion por nodo (botellas)"); ax[0, 0].set_xlabel("nodo")

    inv = pr.groupby("nodo")[["inv_terminado"]].sum()
    inv["inv_sin_etiq"] = vi.groupby("nodo")["inv_sin_etiq"].sum()
    inv.plot.bar(ax=ax[0, 1], stacked=True, color=["tab:green", "tab:olive"])
    ax[0, 1].set_title("Inventario al final del nodo (botellas)"); ax[0, 1].set_xlabel("nodo")

    pr.pivot_table(index="nodo", columns=["vino", "etiqueta"], values="atraso").plot.bar(ax=ax[1, 0], stacked=True)
    ax[1, 0].set_title("Atraso por producto (vino, etiqueta)"); ax[1, 0].set_xlabel("nodo")

    ax[1, 1].bar(nodos, nd["horas"], color="tab:blue")
    ax[1, 1].axhline(res["params"]["T_prod"], color="r", ls="--", label="T_prod")
    ax[1, 1].set_title("Horas de linea usadas"); ax[1, 1].set_xlabel("nodo"); ax[1, 1].set_xticks(nodos)
    ax[1, 1].legend()
    plt.tight_layout()
    return fig


def graficar_comparacion(tabla):
    """Barras apiladas del costo esperado por politica."""
    fig, ax = plt.subplots(figsize=(9, 5))
    tabla[["setups", "inventario", "atrasos"]].plot.bar(ax=ax, stacked=True)
    ax.set_ylabel("Costo esperado ($)"); ax.set_title("Comparacion de politicas")
    plt.xticks(rotation=20); plt.tight_layout()
    return fig
