"""
Horizonte rodante: en vez de resolver todo el horizonte de una vez, en cada nodo se resuelve solo una ventana de
`ventana` periodos hacia adelante (el subarbol que cuelga de ese nodo), se aplica la decision de ese nodo y se pasa
al siguiente con el estado real (inventarios, atrasos y produccion en transito).

  * El costo de la politica se evalua sobre todo el arbol: cada nodo toma su decision con la ventana que parte en
    el, y el costo esperado es sum(pi_n * costo_n), igual que la funcion objetivo del modelo completo.
  * Los productos postergables (zpos) se deciden una sola vez, en la ventana del periodo 1, y quedan fijos
    (en el modelo completo tambien se deciden una vez para todo el horizonte).
  * Con ventana >= numero de periodos la politica coincide con resolver el modelo completo.

Uso:
    python horizonte.py        # corre la grilla y deja CSV + figuras en resultados/horizonte/
"""
import os
import time
import types

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import escalabilidad
import modelo

SALIDA = os.path.join("resultados", "horizonte")


def _subarbol(inst, raiz, niveles):
    """Instancia con el subarbol que cuelga de `raiz` hasta `niveles` periodos (raiz incluida).
    Devuelve (instancia, {id nuevo: id original})."""
    hijos = {n: [m for m in inst.PADRE if inst.PADRE[m] == n] for n in inst.PADRE}
    orden, nivel = [raiz], {raiz: 1}
    for n in orden:                                    # recorrido por niveles
        if nivel[n] < niveles:
            for h in hijos[n]:
                nivel[h] = nivel[n] + 1
                orden.append(h)
    nuevo = {viejo: k for k, viejo in enumerate(orden, start=1)}
    padre = {nuevo[n]: (None if n == raiz else nuevo[inst.PADRE[n]]) for n in orden}
    prob = {}
    for n in orden:                                    # probabilidades condicionales a estar en `raiz`
        prob[nuevo[n]] = 1.0 if n == raiz else prob[nuevo[inst.PADRE[n]]] * inst.PROB[n] / inst.PROB[inst.PADRE[n]]
    dem = {(i, j, nuevo[n]): inst.DEM[i, j, n] for n in orden for i, J in inst.ETIQUETAS.items() for j in J}
    sub = types.SimpleNamespace(PADRE=padre, PROB=prob, DEM=dem, ETIQUETAS=inst.ETIQUETAS, PARAMS=inst.PARAMS)
    return sub, {k: v for v, k in nuevo.items()}


def politica_rodante(inst, PL, ventana=3, tiempo_max=60, verbose=False, **kw):
    """Aplica la politica de horizonte rodante sobre todo el arbol de `inst`. Devuelve un dict con el costo esperado,
    sus componentes, el servicio, los tiempos y el estado de cada ventana."""
    P = inst.PARAMS
    IJ = [(i, j) for i, J in inst.ETIQUETAS.items() for j in J]
    I = list(inst.ETIQUETAS)
    raiz = [n for n in inst.PADRE if inst.PADRE[n] is None][0]
    periodo = modelo._periodos(inst.PADRE)
    n_per = max(periodo.values())

    salida = {}                                        # nodo -> estado al cierre del nodo (para sus hijos)
    zpos = None
    costos = dict(setups=0.0, inventario=0.0, atrasos=0.0)
    no_atendida = demanda = 0.0
    inv_sum = 0.0
    tiempos, estados, brechas = [], [], []
    t0 = time.perf_counter()
    for n in sorted(inst.PADRE):                       # los padres tienen id menor que sus hijos
        sub, viejo = _subarbol(inst, n, min(ventana, n_per - periodo[n] + 1))
        if n == raiz:
            ini = None
        else:
            s = salida[inst.PADRE[n]]
            ini = dict(sb=s["sb"], sbl=s["sbl"], bbl=s["bbl"], wb=s["wb"], wbl=s["wbl"])
        fijar = None if zpos is None else dict(zpos=zpos)
        r = modelo.resolver(PL=PL, datos_mod=sub, tiempo_max=tiempo_max, verbose=verbose, fijar=fijar,
                            estado_inicial=ini, **kw)
        if n == raiz:
            zpos = {ij: int(ij in r["postergables"]) for ij in IJ}
        tiempos.append(r["tamano"]["Tiempo resolución (s)"])
        estados.append(r["status"])
        brechas.append(r["tamano"]["Brecha MIP"])
        # decision y costo del nodo n (el nodo raiz de la ventana)
        pr = r["productos"].query("nodo == 1").set_index(["vino", "etiqueta"])
        vi = r["vinos"].query("nodo == 1").set_index("vino")
        salida[n] = dict(
            sb={i: vi.loc[i, "inv_sin_etiq"] for i in I},
            sbl={ij: pr.loc[ij, "inv_terminado"] for ij in IJ},
            bbl={ij: pr.loc[ij, "atraso"] for ij in IJ},
            wb={i: vi.loc[i, "embotella_sin_etiq"] for i in I},
            wbl={ij: pr.loc[ij, "embotella_etiq"] for ij in IJ})
        pi = inst.PROB[n]
        costos["setups"] += pi * (sum(P["Cz_b"] * vi.loc[i, "setup_b"] for i in I)
                                  + sum(P["Cz_bl"] * pr.loc[ij, "setup_bl"] + P["Cz_l"] * pr.loc[ij, "setup_l"]
                                        for ij in IJ))
        costos["inventario"] += pi * (sum(P["Cs_b"] * salida[n]["sb"][i] for i in I)
                                      + sum(P["Cs_bl"] * salida[n]["sbl"][ij] for ij in IJ))
        costos["atrasos"] += pi * sum(P["Cb_bl"] * salida[n]["bbl"][ij] for ij in IJ)
        demanda += pi * sum(inst.DEM[i, j, n] for i, j in IJ)
        no_atendida += pi * sum(min(inst.DEM[i, j, n], salida[n]["bbl"][i, j]) for i, j in IJ)
        inv_sum += pi * (sum(salida[n]["sb"].values()) + sum(salida[n]["sbl"].values()))
    total = time.perf_counter() - t0
    return dict(costo=sum(costos.values()), **costos, servicio=1 - no_atendida / demanda,
                inventario_prom=inv_sum / n_per, ventanas=len(tiempos), t_total_s=total,
                t_resolucion_total_s=sum(tiempos), t_max_ventana_s=max(tiempos),
                brecha_max=max(brechas), ventanas_limite_tiempo=sum(e != "Optimal" for e in estados),
                postergables=" ".join(f"V{i}E{j}" for (i, j), z in zpos.items() if z) or "-")


def correr(vinos=(2, 3, 4), periodos=(4, 5), ventanas=(2, 3), PLs=("0", "todos"), tiempo_max=60, salida=None, **kw):
    filas = []
    for nv in vinos:
        for npd in periodos:
            inst = escalabilidad.construir_instancia(nv, npd)
            d = escalabilidad.describir(inst)
            for W in ventanas:
                for pl in PLs:
                    PL = d["productos"] if pl == "todos" else int(pl)
                    r = politica_rodante(inst, PL, W, tiempo_max, **kw)
                    filas.append(dict(vinos=nv, productos=d["productos"], periodos=npd, nodos=d["nodos"], ventana=W,
                                      PL=PL, politica=pl, **r))
                    print(f"vinos={nv} periodos={npd} ventana={W} PL={PL}: costo={r['costo']:.0f}, "
                          f"{r['t_total_s']:.1f} s ({r['ventanas']} ventanas, max {r['t_max_ventana_s']:.1f} s, "
                          f"limite en {r['ventanas_limite_tiempo']})", flush=True)
                    if salida:
                        pd.DataFrame(filas).to_csv(salida, index=False, encoding="utf-8-sig")
    return pd.DataFrame(filas)


def comparar_con_completo(h, e):
    """Une los resultados del horizonte rodante (h) con los del modelo completo (e, de escalabilidad.py).
    Agrega las columnas del modelo completo (sufijo _completo) y la diferencia relativa de costo."""
    c = e.rename(columns={"costo": "costo_completo", "estado": "estado_completo", "brecha": "brecha_completo",
                          "t_resolucion_s": "t_completo_s"})
    c = c[["vinos", "periodos", "PL", "costo_completo", "estado_completo", "brecha_completo", "t_completo_s"]]
    m = h.merge(c, on=["vinos", "periodos", "PL"], how="left")
    m["dif_costo_vs_completo"] = m["costo"] / m["costo_completo"] - 1
    # cota inferior del optimo del modelo completo (costo hallado x (1 - brecha)); permite acotar cuanto se pierde
    m["cota_inferior_completo"] = m["costo_completo"] * (1 - m["brecha_completo"])
    m["max_sobrecosto_vs_optimo"] = m["costo"] / m["cota_inferior_completo"] - 1
    return m


_COL_VENTANA = {"completo": "#c9c8c2", 3: modelo.PALETA[0], 2: modelo.PALETA[1]}


def _grupos(m):
    return [(v, p) for p in sorted(m["periodos"].unique()) for v in sorted(m["vinos"].unique())]


def graficar_costos(m):
    """Costo esperado de la politica hibrida (todos los productos postergables): modelo completo contra horizonte
    rodante con ventana 3 y 2. Una columna por numero de periodos."""
    t = m[m["politica"] == "todos"]
    pers = sorted(t["periodos"].unique())
    vinos = sorted(t["vinos"].unique())
    fig, ax = plt.subplots(1, len(pers), figsize=(8.5 * len(pers), 6.0), squeeze=False)
    series = [("completo", "Modelo completo"), (3, "Horizonte rodante, ventana 3"), (2, "Horizonte rodante, ventana 2")]
    w = 0.8 / len(series)
    for a, p in zip(ax[0], pers):
        for k, (clave, nombre) in enumerate(series):
            for xi, v in enumerate(vinos):
                f = t[(t["periodos"] == p) & (t["vinos"] == v)]
                if clave == "completo":
                    r = f.iloc[0]
                    y, lim, txt = r["costo_completo"], r["estado_completo"] != "Optimal", None
                    txt = f"brecha {modelo._pct(r['brecha_completo'])}" if lim else None
                else:
                    r = f[f["ventana"] == clave].iloc[0]
                    y, lim, txt = r["costo"], False, None
                pos = xi + (k - (len(series) - 1) / 2) * w
                a.bar(pos, y, width=w, color=_COL_VENTANA[clave], edgecolor=modelo.SUPERFICIE, linewidth=1,
                      hatch="///" if lim else None, label=nombre if xi == 0 else None)
                a.annotate("$" + modelo._corto(y) + (f"\n{txt}" if txt else ""), (pos, y), ha="center", va="bottom",
                           fontsize=7.5, xytext=(0, 2), textcoords="offset points", linespacing=1.3)
        a.set_xticks(range(len(vinos)), [f"{v} vinos" for v in vinos])
        a.grid(axis="x", visible=False)
        a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "$" + modelo._corto(v)))
        a.margins(y=0.15)
        a.set_title(f"{p} períodos ({int(t[t['periodos'] == p]['nodos'].iloc[0])} nodos)")
    ax[0][0].set_ylabel("Costo esperado")
    h, l = ax[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.01, 0.885), ncol=3)
    fig.suptitle("Política híbrida (todos los productos postergables): modelo completo vs. horizonte rodante\n"
                 "(rayado = el modelo completo alcanzó el límite de tiempo; su costo es el de la mejor solución hallada)",
                 x=0.01, y=0.99, ha="left", fontsize=13, fontweight="bold", linespacing=1.5)
    plt.tight_layout(rect=(0, 0, 1, 0.84))
    return fig


def graficar_tiempos(m, limite=180):
    """Tiempo total de resolucion de la politica hibrida: modelo completo contra horizonte rodante (escala log)."""
    t = m[m["politica"] == "todos"]
    grupos = _grupos(t)
    series = [("completo", "Modelo completo"), (3, "Horizonte rodante, ventana 3"), (2, "Horizonte rodante, ventana 2")]
    w = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(13, 5.6))
    for k, (clave, nombre) in enumerate(series):
        for xi, (v, p) in enumerate(grupos):
            f = t[(t["periodos"] == p) & (t["vinos"] == v)]
            if clave == "completo":
                r = f.iloc[0]
                y, lim = r["t_completo_s"], r["estado_completo"] != "Optimal"
            else:
                y, lim = f[f["ventana"] == clave].iloc[0]["t_total_s"], False
            pos = xi + (k - (len(series) - 1) / 2) * w
            ax.bar(pos, y, width=w, color=_COL_VENTANA[clave], edgecolor=modelo.SUPERFICIE, linewidth=1,
                   hatch="///" if lim else None, label=nombre if xi == 0 else None)
            ax.annotate(f"{modelo._num(y, 1 if y < 10 else 0)} s" + ("\n(límite)" if lim else ""), (pos, y),
                        ha="center", va="bottom", fontsize=8, xytext=(0, 2), textcoords="offset points", linespacing=1.2)
    ax.set_yscale("log")
    ax.set_ylim(top=limite * 6)
    ax.yaxis.set_major_locator(plt.LogLocator(base=10, subs=(1, 3)))
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{modelo._num(v, 1 if v < 1 else 0)} s"))
    ax.set_xticks(range(len(grupos)), [f"{v} vinos\n{p} períodos" for v, p in grupos])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Tiempo total de resolución (escala log)")
    ax.legend(loc="upper left", ncol=3)
    ax.set_title("Tiempo de resolución, política híbrida: modelo completo vs. horizonte rodante\n"
                 "(rayado = el modelo completo se detuvo por el límite de tiempo)", linespacing=1.5)
    plt.tight_layout()
    return fig


def correr_todo(salida=SALIDA, ruta_escalabilidad=None, largos=True, **grilla):
    """Grilla principal, comparacion con el modelo completo (si existe escalabilidad.csv) y horizontes largos.
    `grilla` se pasa a correr() (vinos, periodos, ventanas, PLs, tiempo_max)."""
    os.makedirs(salida, exist_ok=True)
    df = correr(salida=os.path.join(salida, "horizonte_avance.csv"), **grilla)
    df.to_csv(os.path.join(salida, "horizonte.csv"), index=False, encoding="utf-8-sig")
    ruta = ruta_escalabilidad or os.path.join(escalabilidad.SALIDA, "escalabilidad.csv")   # requiere escalabilidad.py
    m = None
    if os.path.exists(ruta):
        m = comparar_con_completo(df, pd.read_csv(ruta))
        m.to_csv(os.path.join(salida, "horizonte_vs_completo.csv"), index=False, encoding="utf-8-sig")
        for nombre, fig in (("comparacion_costos", graficar_costos(m)), ("comparacion_tiempos", graficar_tiempos(m))):
            modelo.exportar_figura(fig, nombre, carpeta=salida, dpi=300)
            plt.close(fig)
    else:
        print("No se encontro", ruta, "- correr escalabilidad.py para generar la comparacion con el modelo completo.")
    if largos:   # horizontes que el modelo completo ya no resuelve (solo ventana 2)
        largo = correr(vinos=(2, 4), periodos=(6, 7), ventanas=(2,),
                       salida=os.path.join(salida, "horizonte_periodos_largos.csv"))
        largo.to_csv(os.path.join(salida, "horizonte_periodos_largos.csv"), index=False, encoding="utf-8-sig")
    return df, m


if __name__ == "__main__":
    correr_todo()
