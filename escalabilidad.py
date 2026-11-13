"""
Escalabilidad: que pasa con mas vinos (3 o 4) y mas periodos (4 o 5), en resultados y en tiempos.

Instancias sinteticas, construidas a partir de la instancia base (no hay datos reales de otros vinos):
  * Vinos nuevos: replicas de los vinos existentes (el vino 3 copia al vino 1, el vino 4 al vino 2), con la
    demanda multiplicada por un factor (0,9 y 1,1) para que no sean identicas y se rompa la simetria.
  * Periodos nuevos: el arbol se extiende como una cadena de Markov de 3 estados (alto / medio / bajo).
    Los hijos de un nodo copian a los hijos del nodo del periodo 2 que tiene su mismo estado:
        alto  -> nodos 5, 6      (probabilidades condicionales 0,7 y 0,3)
        medio -> nodos 7, 8, 9   (1/3 cada uno)
        bajo  -> nodos 10, 11    (0,6 y 0,4)
  * Capacidad: con escala_capacidad=True las horas de linea y los estanques crecen en proporcion a la demanda
    esperada, para mantener la holgura de la instancia base y aislar el efecto del tamaño. Con False se
    mantienen 84 h y 20 estanques.

Uso:
    python escalabilidad.py            # corre la grilla y deja CSV + figuras en resultados/escalabilidad/
"""
import math
import os
import types

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datos
import modelo

SALIDA = os.path.join("resultados", "escalabilidad")

# estado de cada nodo del periodo 2 y 3 y plantilla de hijos de cada estado (nodos del arbol base)
_ESTADO = {2: "A", 3: "M", 4: "B", 5: "A", 6: "M", 7: "A", 8: "M", 9: "B", 10: "M", 11: "B"}
_PLANTILLA = {"A": [5, 6], "M": [7, 8, 9], "B": [10, 11]}
_PLANTILLA_PERIODO2 = {"A": 2, "M": 3, "B": 4}
# vino nuevo k (3, 4, ...) -> (vino que replica, factor de demanda)
_REPLICAS = {3: (1, 0.9), 4: (2, 1.1), 5: (1, 1.2), 6: (2, 0.8)}


def construir_instancia(n_vinos=2, n_periodos=3, escala_capacidad=True, base=datos):
    """Devuelve un objeto con PADRE, PROB, DEM, ETIQUETAS, PARAMS (el mismo formato que datos.py)."""
    if n_periodos < 3:
        raise ValueError("n_periodos debe ser >= 3")
    # --- arbol: nodo -> (padre, prob. condicional, nodo plantilla del que copia la demanda)
    padre = dict(base.PADRE)
    cond = {n: base.PROB[n] / (base.PROB[base.PADRE[n]] if base.PADRE[n] else 1.0) for n in base.PADRE}
    plantilla = {n: n for n in base.PADRE}
    estado = dict(_ESTADO)
    periodo = {n: p for n, p in modelo._periodos(base.PADRE).items()}
    hojas = [n for n in base.PADRE if not any(base.PADRE[m] == n for m in base.PADRE)]
    sig = max(base.PADRE) + 1
    for p in range(4, n_periodos + 1):
        nuevas = []
        for h in hojas:
            for tpl in _PLANTILLA[estado[h]]:
                padre[sig], cond[sig], plantilla[sig], estado[sig], periodo[sig] = h, cond[tpl], tpl, estado[tpl], p
                nuevas.append(sig)
                sig += 1
        hojas = nuevas
    prob = {}
    for n in sorted(padre):
        prob[n] = cond[n] * (prob[padre[n]] if padre[n] else 1.0)

    # --- vinos y demanda
    etiquetas = dict(base.ETIQUETAS)
    origen = {i: (i, 1.0) for i in etiquetas}
    for k in range(3, n_vinos + 1):
        v, f = _REPLICAS[k]
        etiquetas[k] = list(base.ETIQUETAS[v])
        origen[k] = (v, f)
    dem = {}
    for n in padre:
        for i, J in etiquetas.items():
            v, f = origen[i]
            for j in J:
                dem[i, j, n] = f * base.DEM[v, j, plantilla[n]]

    # --- parametros (capacidad proporcional a la demanda esperada por periodo)
    P = dict(base.PARAMS)
    if n_vinos > 2 and escala_capacidad:
        tot = lambda D, pr, nodos: sum(pr[n] * sum(D[k] for k in D if k[2] == n) for n in nodos)
        # solo periodos 2 y 3, para que la capacidad dependa del numero de vinos y no del horizonte
        n_ref = [n for n in padre if periodo[n] in (2, 3)]
        ref = tot(base.DEM, base.PROB, n_ref) / sum(base.PROB[n] for n in n_ref)
        nuevo = tot(dem, prob, n_ref) / sum(prob[n] for n in n_ref)
        r = nuevo / ref
        P["T_prod"] = round(base.PARAMS["T_prod"] * r, 1)
        P["K_tank"] = math.ceil(base.PARAMS["K_tank"] * r)
        P["M_b"] = P["M_bl"] = P["K_tank"] * P["V_tank"] / P["V_bot"]
        P["M_l"] = (P["T_prod"] - P["Tz_l"]) / P["Tw_l"]
    return types.SimpleNamespace(PADRE=padre, PROB=prob, DEM=dem, ETIQUETAS=etiquetas, PARAMS=P,
                                 N_VINOS=n_vinos, N_PERIODOS=n_periodos)


def describir(inst):
    hojas = [n for n in inst.PADRE if not any(inst.PADRE[m] == n for m in inst.PADRE)]
    return dict(vinos=len(inst.ETIQUETAS), productos=sum(len(J) for J in inst.ETIQUETAS.values()),
                periodos=inst.N_PERIODOS, nodos=len(inst.PADRE), hojas=len(hojas),
                T_prod=inst.PARAMS["T_prod"], K_tank=inst.PARAMS["K_tank"],
                suma_prob_hojas=sum(inst.PROB[n] for n in hojas))


# ---------------------------------------------------------------------------
# Experimento
# ---------------------------------------------------------------------------
def correr(vinos=(2, 3, 4), periodos=(3, 4, 5), PLs=("0", "todos"), tiempo_max=180, escala_capacidad=True,
           salida=None, **kw):
    """Resuelve cada combinacion (vinos, periodos, PL). PL='todos' deja postergable cualquier producto.
    Si se pasa `salida` (ruta de CSV), guarda el avance despues de cada corrida."""
    filas = []
    for nv in vinos:
        for npd in periodos:
            inst = construir_instancia(nv, npd, escala_capacidad)
            d = describir(inst)
            vistos = set()
            for pl in PLs:
                PL = d["productos"] if pl == "todos" else int(pl)
                if PL in vistos:
                    continue
                vistos.add(PL)
                r = modelo.resolver(PL=PL, datos_mod=inst, tiempo_max=tiempo_max, **kw)
                t = r["tamano"]
                filas.append(dict(**d, PL=PL, politica=pl, estado=r["status"], costo=r["obj"],
                                  brecha=t["Brecha MIP"], variables=t["Variables"],
                                  binarias_enteras=t["Variables enteras/binarias"], restricciones=t["Restricciones"],
                                  t_construccion_s=t["Tiempo construcción (s)"], t_resolucion_s=t["Tiempo resolución (s)"],
                                  setups=r["kpis"]["Set-ups (total)"], servicio=r["kpis"]["Nivel de servicio"],
                                  util_max=r["kpis"]["Utilización máxima"], util_media=r["kpis"]["Utilización media"],
                                  atraso_prom=r["kpis"]["Atraso promedio (botellas)"],
                                  inventario_prom=r["kpis"]["Inventario promedio (total)"],
                                  postergables=" ".join(f"V{i}E{j}" for i, j in r["postergables"]) or "-"))
                print(f"vinos={nv} periodos={npd} nodos={d['nodos']} PL={PL}: {r['status']}, "
                      f"costo={r['obj']:.0f}, vars={t['Variables']}, {t['Tiempo resolución (s)']:.1f} s, "
                      f"brecha={t['Brecha MIP']:.2%}", flush=True)
                if salida:
                    pd.DataFrame(filas).to_csv(salida, index=False, encoding="utf-8-sig")
    df = pd.DataFrame(filas)
    base = df[df["PL"] == 0].set_index(["vinos", "periodos"])["costo"]
    df["ahorro_vs_a_stock"] = [1 - c / base[(v, p)] if (v, p) in base.index and base[(v, p)] > 0 else np.nan
                               for c, v, p in zip(df["costo"], df["vinos"], df["periodos"])]
    return df


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def _colores_vinos(vinos):
    return dict(zip(vinos, [modelo.PALETA[0], modelo.PALETA[1], modelo.PALETA[2], modelo.PALETA[4]]))


def _eje_log_dinero(a):
    a.yaxis.set_major_locator(plt.LogLocator(base=10, subs=(1, 2, 5)))
    a.yaxis.set_minor_formatter(plt.NullFormatter())
    a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "$" + modelo._corto(v)))


def graficar_tiempos(df, pl="todos", limite=None):
    """Izquierda: tiempo de resolucion contra numero de nodos del arbol (log). Derecha: variables del modelo.
    Los puntos huecos son corridas que se detuvieron por limite de tiempo (con su brecha)."""
    t = df[df["politica"] == pl].sort_values(["vinos", "nodos"])
    col = _colores_vinos(sorted(t["vinos"].unique()))
    nodos = sorted(t["nodos"].unique())
    limite = limite or t.loc[t["estado"] != "Optimal", "t_resolucion_s"].min()
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.6))
    for k, (v, g) in enumerate(t.groupby("vinos")):
        etiqueta = f"{v} vinos ({int(g['productos'].iloc[0])} productos)"
        ax[0].plot(g["nodos"], g["t_resolucion_s"].clip(lower=0.01), color=col[v], lw=2, label=etiqueta, zorder=2)
        ax[1].plot(g["nodos"], g["variables"], color=col[v], lw=2, label=etiqueta, zorder=2)
        for _, r in g.iterrows():
            lleno = r["estado"] == "Optimal"
            ax[0].scatter(r["nodos"], r["t_resolucion_s"], s=70, zorder=3, color=col[v] if lleno else "white",
                          edgecolor=col[v], linewidth=2)
            ax[1].scatter(r["nodos"], r["variables"], s=70, zorder=3, color=col[v], edgecolor=col[v])
            if not lleno:
                ax[0].annotate(f"{modelo._pct(r['brecha'])}", (r["nodos"], r["t_resolucion_s"]), fontsize=9,
                               fontweight="bold", color=col[v], xytext=(0, 9 + 11 * k), textcoords="offset points",
                               ha="center")
    ax[0].set_ylim(top=limite * 5)   # espacio sobre el limite para los rotulos de brecha
    ax[0].axhline(limite, color=modelo.TEXTO_2, lw=1, ls="--")
    ax[0].annotate(f"límite de tiempo ({modelo._num(limite)} s)", (nodos[0], limite), fontsize=9,
                   color=modelo.TEXTO_2, xytext=(0, 4), textcoords="offset points")
    ax[0].set_yscale("log")
    ax[0].yaxis.set_major_locator(plt.LogLocator(base=10, subs=(1, 3)))
    ax[0].yaxis.set_minor_formatter(plt.NullFormatter())
    ax[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: modelo._num(v, 1 if v < 1 else 0)))
    ax[0].set_ylabel("Tiempo de resolución (s, escala log)")
    ax[0].set_title("Tiempo de resolución\n(punto hueco = límite de tiempo, % = brecha)", linespacing=1.5)
    ax[0].legend(loc="upper left", bbox_to_anchor=(0, 0.88))
    ax[1].set_yscale("log")
    ax[1].yaxis.set_major_locator(plt.LogLocator(base=10, subs=(1, 2, 5)))
    ax[1].yaxis.set_minor_formatter(plt.NullFormatter())
    ax[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: modelo._num(v)))
    ax[1].set_ylabel("Variables (escala log)")
    ax[1].set_title("Tamaño del modelo\n", linespacing=1.5)
    for a in ax:
        a.set_xscale("log")
        a.set_xticks(nodos, [modelo._num(n) for n in nodos])
        a.xaxis.set_minor_formatter(plt.NullFormatter())
        a.set_xlabel("Nodos del árbol de escenarios (3, 4 y 5 períodos)")
        a.grid(axis="x", visible=True)
    nombre = "todos los productos postergables" if pl == "todos" else f"PL = {pl}"
    fig.suptitle(f"Crecimiento del modelo y tiempo de resolución ({nombre})", x=0.01, ha="left", fontsize=14,
                 fontweight="bold")
    plt.tight_layout()
    return fig


def graficar_resultados(df):
    """Ahorro de la postergacion frente a A stock (izq.) y costo esperado de A stock e hibrida total (der.).
    Las barras rayadas y los puntos huecos son corridas que se detuvieron por limite de tiempo: el costo es el de la
    mejor solucion encontrada (cota superior del optimo), asi que el ahorro real es al menos ese."""
    pers = sorted(df["periodos"].unique())
    vinos = sorted(df["vinos"].unique())
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.8))
    x = np.arange(len(pers))
    w = 0.8 / len(vinos)
    col = _colores_vinos(vinos)
    h = df[df["politica"] == "todos"]
    minimo = 0
    for k, v in enumerate(vinos):
        s = h[h["vinos"] == v].set_index("periodos")
        for xi, p in zip(x + (k - (len(vinos) - 1) / 2) * w, pers):
            if p not in s.index:
                continue
            r = s.loc[p]
            y = 100 * r["ahorro_vs_a_stock"]
            minimo = min(minimo, y)
            lim = r["estado"] != "Optimal"
            ax[0].bar(xi, y, width=w, color=col[v], edgecolor="white" if not lim else modelo.SUPERFICIE,
                      linewidth=1, hatch="///" if lim else None, alpha=0.55 if lim else 1,
                      label=f"{v} vinos" if p == pers[0] else None)
            txt = f"{y:.0f} %" + (f"\nbrecha {modelo._pct(r['brecha'])}" if lim else "")
            # los ahorros negativos se rotulan sobre la linea de cero para no pisar el eje x
            ax[0].annotate(txt, (xi, max(y, 0)), ha="center", va="bottom", fontsize=8.5,
                           xytext=(0, 2), textcoords="offset points", linespacing=1.3)
    ax[0].axhline(0, color=modelo.TEXTO_2, lw=1)
    ax[0].set_xticks(x, [f"{p} períodos" for p in pers]); ax[0].set_ylim(min(-8, minimo * 1.6), 112)
    ax[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f} %"))
    ax[0].set_title("Ahorro de postergar (todos los productos) frente a A stock\n(rayado = límite de tiempo)",
                    linespacing=1.5)
    ax[0].grid(axis="x", visible=False); ax[0].legend(loc="upper right", ncol=3)
    for pol, estilo in (("0", "--"), ("todos", "-")):
        for v in vinos:
            s = df[(df["politica"] == pol) & (df["vinos"] == v)].sort_values("periodos")
            ax[1].plot(s["periodos"], s["costo"], ls=estilo, color=col[v], lw=2, zorder=2,
                       label=f"{v} vinos" if pol == "0" else None)
            for _, r in s.iterrows():
                lleno = r["estado"] == "Optimal"
                ax[1].scatter(r["periodos"], r["costo"], s=60, zorder=3, color=col[v] if lleno else "white",
                              edgecolor=col[v], linewidth=2)
    ax[1].set_yscale("log"); ax[1].set_xticks(pers)
    _eje_log_dinero(ax[1])
    ax[1].set_xlabel("Períodos")
    ax[1].set_title("Costo esperado total\n(punteado: A stock; continuo: postergar todo; hueco: límite de tiempo)",
                    linespacing=1.5, fontsize=10.5)
    ax[1].legend(loc="upper left")
    fig.suptitle("Resultados al agregar vinos y períodos", x=0.01, ha="left", fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def correr_todo(**kw):
    os.makedirs(SALIDA, exist_ok=True)
    df = correr(salida=os.path.join(SALIDA, "escalabilidad_avance.csv"), **kw)
    df.to_csv(os.path.join(SALIDA, "escalabilidad.csv"), index=False, encoding="utf-8-sig")
    for nombre, fig in (("tiempos", graficar_tiempos(df)), ("resultados", graficar_resultados(df))):
        modelo.exportar_figura(fig, nombre, carpeta=SALIDA, dpi=300)
        plt.close(fig)
    return df


if __name__ == "__main__":
    correr_todo()
