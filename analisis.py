"""
Analisis de la Entrega 3: cotas VSS/EVPI, sensibilidad al costo de atraso y valor residual.

Uso:
    python analisis.py            # corre todo y deja CSV + PNG en resultados/
    import analisis; analisis.cotas(PL=5)
"""
import os
import types

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import datos
import modelo

SALIDA = "resultados"
POLITICAS = (0, 2, 3, 4, 5)


# ---------------------------------------------------------------------------
# Instancias auxiliares (cadenas deterministas) para EV y WS
# ---------------------------------------------------------------------------
def _cadena(demanda_por_periodo, base=datos):
    """Instancia de 3 nodos en cadena (1 -> 2 -> 3), probabilidad 1 y demanda dada por periodo."""
    n_per = len(demanda_por_periodo)
    return types.SimpleNamespace(
        PADRE={k: (None if k == 1 else k - 1) for k in range(1, n_per + 1)},
        PROB={k: 1.0 for k in range(1, n_per + 1)},
        DEM={(i, j, k): demanda_por_periodo[k - 1][i, j] for k in range(1, n_per + 1)
             for i in base.ETIQUETAS for j in base.ETIQUETAS[i]},
        ETIQUETAS=base.ETIQUETAS, PARAMS=base.PARAMS)


def _instancia_ev(base=datos):
    """Demanda esperada de cada periodo (el periodo 1 es conocido)."""
    per = modelo._periodos(base.PADRE)
    dem = []
    for p in range(1, max(per.values()) + 1):
        dem.append({(i, j): sum(base.PROB[n] * base.DEM[i, j, n] for n in base.PADRE if per[n] == p)
                    for i in base.ETIQUETAS for j in base.ETIQUETAS[i]})
    return _cadena(dem, base)


def _trayectorias(base=datos):
    """Una instancia en cadena por cada hoja del arbol, con su probabilidad."""
    hijos = {n: [m for m in base.PADRE if base.PADRE[m] == n] for n in base.PADRE}
    out = {}
    for hoja in (n for n in base.PADRE if not hijos[n]):
        camino, n = [], hoja
        while n is not None:
            camino.append(n)
            n = base.PADRE[n]
        camino.reverse()
        dem = [{(i, j): base.DEM[i, j, n] for i in base.ETIQUETAS for j in base.ETIQUETAS[i]} for n in camino]
        out[hoja] = (camino, base.PROB[hoja], _cadena(dem, base))
    return out


# ---------------------------------------------------------------------------
# Cotas: EV, EEV, WS, VSS, EVPI
# ---------------------------------------------------------------------------
def _decision_periodo1(res):
    """Decision del nodo raiz de una solucion: lo que se fija para el EEV."""
    pr = res["productos"].query("nodo == 1")
    vi = res["vinos"].query("nodo == 1")
    return dict(
        wbl={(int(r.vino), int(r.etiqueta)): r.embotella_etiq for r in pr.itertuples()},
        wl={(int(r.vino), int(r.etiqueta)): r.etiqueta_guardadas for r in pr.itertuples()},
        wb={int(r.vino): r.embotella_sin_etiq for r in vi.itertuples()},
        zpos={ij: int(ij in res["postergables"]) for ij in res["IJ"]})


def cotas(PL=0, **kw):
    """Calcula RP, EV, EEV, WS, VSS y EVPI para una politica PL.
    RP  : optimo del problema estocastico (el modelo de siempre).
    EV  : modelo determinista con la demanda esperada de cada periodo.
    EEV : se fija la decision del periodo 1 (y los postergables) del EV y se reoptimiza el resto en los 11 nodos.
    WS  : cada trayectoria se resuelve con informacion perfecta; se pondera por la probabilidad de la hoja.
    VSS = EEV - RP ; EVPI = RP - WS."""
    rp = modelo.resolver(PL=PL, **kw)
    ev = modelo.resolver(PL=PL, datos_mod=_instancia_ev(), **kw)
    dec = _decision_periodo1(ev)
    eev = modelo.resolver(PL=PL, fijar=dec, **kw)
    # variante menos estricta: solo se fija la produccion del periodo 1, los postergables quedan libres
    eev_libre = modelo.resolver(PL=PL, fijar={k: v for k, v in dec.items() if k != "zpos"}, **kw)

    filas, ws = [], 0.0
    for hoja, (camino, prob, inst) in _trayectorias().items():
        r = modelo.resolver(PL=PL, datos_mod=inst, **kw)
        ws += prob * r["obj"]
        filas.append(dict(PL=PL, hoja=hoja, camino="-".join(map(str, camino)), prob=prob, costo=r["obj"],
                          postergables=" ".join(f"V{i}E{j}" for i, j in r["postergables"]) or "-",
                          estado=r["status"]))
    estados = {rp["status"], ev["status"], eev["status"], eev_libre["status"], *[f["estado"] for f in filas]}
    tol = 1e-6 * rp["obj"]
    resumen = dict(PL=PL, EV=ev["obj"], WS=ws, RP=rp["obj"], EEV=eev["obj"],
                   VSS=eev["obj"] - rp["obj"], EVPI=rp["obj"] - ws,
                   VSS_pct=(eev["obj"] - rp["obj"]) / eev["obj"] if eev["obj"] else np.nan,
                   EVPI_pct=(rp["obj"] - ws) / rp["obj"] if rp["obj"] else np.nan,
                   EEV_postergables_libres=eev_libre["obj"], VSS_postergables_libres=eev_libre["obj"] - rp["obj"],
                   postergables_RP=" ".join(f"V{i}E{j}" for i, j in rp["postergables"]) or "-",
                   postergables_EV=" ".join(f"V{i}E{j}" for i, j in ev["postergables"]) or "-",
                   orden_ok=bool(ws <= rp["obj"] + tol <= eev["obj"] + 2 * tol and ev["obj"] <= ws + tol),
                   estados="/".join(sorted(estados)))
    plan = pd.DataFrame([dict(PL=PL, producto=f"V{i}E{j}",
                              demanda_periodo1=datos.DEM[i, j, 1],
                              produccion_RP=_decision_periodo1(rp)["wbl"][i, j],
                              produccion_EV=dec["wbl"][i, j])
                         for i, j in rp["IJ"]]
                        + [dict(PL=PL, producto=f"V{i} sin etiqueta", demanda_periodo1=np.nan,
                                produccion_RP=_decision_periodo1(rp)["wb"][i], produccion_EV=dec["wb"][i])
                           for i in datos.ETIQUETAS])
    return resumen, pd.DataFrame(filas), plan


def tabla_cotas(PLs=(0, 5)):
    res, ws, plan = [], [], []
    for PL in PLs:
        r, w, p = cotas(PL)
        res.append(r); ws.append(w); plan.append(p)
    return pd.DataFrame(res).set_index("PL"), pd.concat(ws, ignore_index=True), pd.concat(plan, ignore_index=True)


def graficar_cotas(tab):
    """Por politica: WS <= RP <= EEV (y EV como referencia) con VSS y EVPI rotulados."""
    cols = [("WS", "WS (info. perfecta)", "#0f9479"), ("RP", "RP (estocástico)", modelo.VINO),
            ("EEV", "EEV (solución determinista)", "#d1962a"), ("EV", "EV (determinista)", "#c9c8c2")]
    fig, ax = plt.subplots(1, len(tab), figsize=(6.2 * len(tab), 5.6))
    ax = np.atleast_1d(ax)
    for a, (PL, f) in zip(ax, tab.iterrows()):
        vals = [f[c] for c, _, _ in cols]
        a.bar(range(len(cols)), vals, color=[c for _, _, c in cols], width=0.62)
        for k, v in enumerate(vals):
            a.annotate("$" + modelo._corto(v), (k, v), ha="center", va="bottom", fontsize=10, fontweight="bold",
                       xytext=(0, 3), textcoords="offset points")
        a.set_xticks(range(len(cols)), [n.replace(" (", "\n(") for _, n, _ in cols], fontsize=9)
        a.grid(axis="x", visible=False)
        a.set_yscale("log")  # las cotas difieren en órdenes de magnitud
        a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "$" + modelo._corto(v)))
        a.set_ylim(min(vals) / 3, max(vals) * 6)
        a.set_title(f"{modelo._nombre_politica(PL)}\nVSS = \\${modelo._corto(f['VSS'])} ({modelo._pct(f['VSS_pct'], 1)})"
                    f"   |   EVPI = \\${modelo._corto(f['EVPI'])} ({modelo._pct(f['EVPI_pct'], 1)})",
                    fontsize=11, linespacing=1.6)
    ax[0].set_ylabel("Costo esperado (escala log)")
    fig.suptitle("Cotas del valor de la incertidumbre: VSS = EEV − RP, EVPI = RP − WS", x=0.01, ha="left",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sensibilidades (costo de atraso y valor residual)
# ---------------------------------------------------------------------------
def _corridas(parametro, valores, PLs, **extra):
    filas = []
    for v in valores:
        for PL in PLs:
            r = modelo.resolver(PL=PL, **{parametro: v}, **extra)
            filas.append({parametro: v, "PL": PL, "politica": modelo._nombre_politica(PL), "estado": r["status"],
                          "costo": r["obj"], **r["costo_detalle"], **r["kpis"],
                          "postergables": " ".join(f"V{i}E{j}" for i, j in r["postergables"]) or "-",
                          "tiempo_resolucion_s": r["tamano"]["Tiempo resolución (s)"]})
    return pd.DataFrame(filas)


def sens_atraso(valores=(1500, 5000, 15000), PLs=POLITICAS):
    return _corridas("Cb_bl", valores, PLs)


def sens_alfa(valores=(0, 0.5, 0.9), PLs=POLITICAS):
    return _corridas("alfa", valores, PLs)


def plan_inicial_igual(valores=(0, 0.5, 0.9), PLs=(0, 2, 5)):
    """Verifica la afirmacion del PDF: alfa no cambia el plan (produccion de cada nodo ni postergables)."""
    filas = []
    for PL in PLs:
        ref = modelo.resolver(PL=PL, alfa=0)
        for a in valores:
            r = modelo.resolver(PL=PL, alfa=a)
            d = max(np.abs(r["productos"]["embotella_etiq"] - ref["productos"]["embotella_etiq"]).max(),
                    np.abs(r["vinos"]["embotella_sin_etiq"] - ref["vinos"]["embotella_sin_etiq"]).max(),
                    np.abs(r["productos"]["etiqueta_guardadas"] - ref["productos"]["etiqueta_guardadas"]).max())
            filas.append(dict(PL=PL, alfa=a, max_dif_botellas=d, mismos_postergables=r["postergables"] == ref["postergables"]))
    return pd.DataFrame(filas)


def _barras_por_nivel(a, tab, col, niveles, fmt_nivel, fmt_val, pols=POLITICAS):
    x = np.arange(len(niveles))
    w = 0.8 / len(pols)
    for k, PL in enumerate(pols):
        nombre = modelo._nombre_politica(PL)
        v = [tab[(tab["nivel"] == n) & (tab["PL"] == PL)][col].iloc[0] for n in niveles]
        a.bar(x + (k - (len(pols) - 1) / 2) * w, v, width=w, color=modelo._color_pol(nombre), label=nombre,
              edgecolor=modelo.SUPERFICIE, linewidth=1)
    a.set_xticks(x, [fmt_nivel(n) for n in niveles])
    a.grid(axis="x", visible=False)
    a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: fmt_val(v)))


def graficar_sens(df, parametro, fmt_nivel, xlabel, titulo, base):
    """Izquierda: costo esperado por politica. Centro: ahorro vs. A stock. Derecha: nivel de servicio."""
    t = df.rename(columns={parametro: "nivel"}).copy()
    niveles = sorted(t["nivel"].unique())
    pols = [p for p in POLITICAS if p in set(t["PL"])]
    ref = t[t["PL"] == 0].set_index("nivel")["costo"]
    t["ahorro"] = [100 * (1 - c / ref[n]) for c, n in zip(t["costo"], t["nivel"])]
    t["servicio"] = 100 * t["Nivel de servicio"]
    lbl = lambda n: fmt_nivel(n) + ("\n(caso base)" if n == base else "")
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.4))
    _barras_por_nivel(ax[0], t, "costo", niveles, lbl, lambda v: "$" + modelo._corto(v), pols)
    ax[0].set_title("Costo esperado")
    if t["costo"].max() > 20 * t["costo"].min():   # órdenes de magnitud de diferencia: escala log para que se vean todas
        ax[0].set_yscale("log")
        ax[0].set_title("Costo esperado (escala log)")
        ax[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "$" + modelo._corto(v)))
        ax[0].yaxis.set_minor_formatter(plt.NullFormatter())
    t2 = t[t["PL"] > 0]
    _barras_por_nivel(ax[1], t2, "ahorro", niveles, lbl, lambda v: f"{v:.0f} %", [p for p in pols if p > 0])
    ax[1].set_title("Ahorro frente a A stock")
    ax[1].set_ylim(0, 105)
    _barras_por_nivel(ax[2], t, "servicio", niveles, lbl, lambda v: f"{v:.0f} %", pols)
    ax[2].set_title("Nivel de servicio")
    ax[2].set_ylim(0, 105)
    for a in ax:
        a.set_xlabel(xlabel)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.01, 0.95), ncol=len(pols))
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


# ---------------------------------------------------------------------------
# Arboles con otra variabilidad (CV = desviacion estandar / media)
# ---------------------------------------------------------------------------
def cv_por_producto_periodo(base=datos):
    """CV de cada producto y periodo (ponderado por probabilidad, como la Tabla B.1 del informe)."""
    per = modelo._periodos(base.PADRE)
    filas = []
    for p in range(2, max(per.values()) + 1):
        ns = [n for n in base.PADRE if per[n] == p]
        pi = np.array([base.PROB[n] for n in ns])
        for i in base.ETIQUETAS:
            for j in base.ETIQUETAS[i]:
                d = np.array([base.DEM[i, j, n] for n in ns])
                m = (pi * d).sum()
                filas.append(dict(periodo=p, producto=f"V{i}E{j}", media=m,
                                  cv=np.sqrt((pi * (d - m) ** 2).sum()) / m))
    return pd.DataFrame(filas)


def arbol_variabilidad(cv, base=datos):
    """Mismo arbol y probabilidades, pero la desviacion de cada producto y periodo se reescala para que
    su CV sea `cv`. La media de cada producto y periodo no cambia, y el periodo 1 (conocido) tampoco."""
    per = modelo._periodos(base.PADRE)
    dem = dict(base.DEM)
    for p in range(2, max(per.values()) + 1):
        ns = [n for n in base.PADRE if per[n] == p]
        pi = np.array([base.PROB[n] for n in ns])
        for i in base.ETIQUETAS:
            for j in base.ETIQUETAS[i]:
                d = np.array([base.DEM[i, j, n] for n in ns])
                m = (pi * d).sum()
                sd = np.sqrt((pi * (d - m) ** 2).sum())
                nuevo = m + (d - m) * (cv * m / sd)
                if nuevo.min() < 0:
                    raise ValueError(f"CV={cv} da demanda negativa en V{i}E{j}, periodo {p}")
                dem.update({(i, j, n): v for n, v in zip(ns, nuevo)})
    return types.SimpleNamespace(PADRE=base.PADRE, PROB=base.PROB, DEM=dem, ETIQUETAS=base.ETIQUETAS,
                                 PARAMS=base.PARAMS)


def sens_variabilidad(cvs=(0.1, 0.3, 0.5), PLs=POLITICAS, **kw):
    """Costo de cada politica para arboles con CV objetivo `cvs`. El arbol original va como nivel -1."""
    filas = []
    for cv in (-1, *cvs):
        arbol = datos if cv == -1 else arbol_variabilidad(cv)
        for PL in PLs:
            r = modelo.resolver(PL=PL, datos_mod=arbol, **kw)
            filas.append(dict(cv=cv, PL=PL, politica=modelo._nombre_politica(PL), estado=r["status"],
                              costo=r["obj"], **r["costo_detalle"], **r["kpis"],
                              postergables=" ".join(f"V{i}E{j}" for i, j in r["postergables"]) or "-",
                              tiempo_resolucion_s=r["tamano"]["Tiempo resolución (s)"]))
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Arboles con otra correlacion entre las etiquetas de un mismo vino
# ---------------------------------------------------------------------------
def _choques(pi, candidatos, k):
    """k choques estandarizados (media 0, varianza 1, ortogonales, ponderados por pi) construidos por
    Gram-Schmidt a partir de los vectores candidatos (datos del arbol base, para conservar su orden alto/medio/bajo)."""
    ip = lambda a, b: float((pi * a * b).sum())
    base = [np.ones_like(pi) / np.sqrt(ip(np.ones_like(pi), np.ones_like(pi)))]
    for c in candidatos:
        v = np.asarray(c, float).copy()
        for b in base:
            v -= ip(v, b) * b
        if ip(v, v) > 1e-9:
            base.append(v / np.sqrt(ip(v, v)))
        if len(base) == k + 1:
            break
    if len(base) < k + 1:
        raise ValueError("no hay suficientes nodos para generar los choques")
    return base[1:]


def _simplex(k):
    """k vectores unitarios con producto punto -1/(k-1) entre cada par: la correlacion mas negativa posible
    entre k variables con la misma correlacion de a pares (k=2 -> -1, k=3 -> -0,5)."""
    if k == 1:
        return np.ones((1, 1))
    E = np.eye(k) - 1.0 / k
    _, _, vt = np.linalg.svd(E)
    V = E @ vt[: k - 1].T
    return V / np.linalg.norm(V, axis=1, keepdims=True)


def arbol_correlacion(cv, signo, base=datos):
    """Arbol con CV = cv en todos los productos (periodos 2 y 3), misma media y probabilidades que el base, y
    correlacion entre las etiquetas de cada vino segun `signo` = {vino: '+' | '-'} (o '+' / '-' para ambos):
      '+': todas las etiquetas del vino suben y bajan juntas (correlacion +1): D = m (1 + cv z).
      '-': correlacion negativa equicorrelada (-1 con 2 etiquetas, -0,5 con 3); la demanda total del vino
           varia poco y lo que cambia es la mezcla entre etiquetas."""
    if isinstance(signo, str):
        signo = {i: signo for i in base.ETIQUETAS}
    per = modelo._periodos(base.PADRE)
    dem = dict(base.DEM)
    for p in range(2, max(per.values()) + 1):
        ns = [n for n in base.PADRE if per[n] == p]
        pi = np.array([base.PROB[n] for n in ns])
        total = np.array([sum(base.DEM[i, j, n] for i in base.ETIQUETAS for j in base.ETIQUETAS[i]) for n in ns])
        for i, J in base.ETIQUETAS.items():
            k = len(J)
            cand = [total] + [np.array([base.DEM[i, j, n] for n in ns]) for j in J]
            if signo[i] == "+":
                e = {j: _choques(pi, cand, 1)[0] for j in J}
            else:
                u = _choques(pi, cand, k - 1)
                V = _simplex(k)
                e = {j: sum(V[r, c] * u[c] for c in range(k - 1)) for r, j in enumerate(J)}
            for j in J:
                m = sum(base.PROB[n] * base.DEM[i, j, n] for n in ns)
                nuevo = m * (1 + cv * e[j])
                if nuevo.min() < 0:
                    raise ValueError(f"CV={cv} y signo {signo[i]} dan demanda negativa en V{i}E{j}, periodo {p}")
                dem.update({(i, j, n): v for n, v in zip(ns, nuevo)})
    return types.SimpleNamespace(PADRE=base.PADRE, PROB=base.PROB, DEM=dem, ETIQUETAS=base.ETIQUETAS,
                                 PARAMS=base.PARAMS)


def estadisticas_arbol(arbol):
    """CV y correlacion media entre las etiquetas de cada vino, por periodo (lo realmente logrado)."""
    per = modelo._periodos(arbol.PADRE)
    filas = []
    for p in range(2, max(per.values()) + 1):
        ns = [n for n in arbol.PADRE if per[n] == p]
        pi = np.array([arbol.PROB[n] for n in ns])
        for i, J in arbol.ETIQUETAS.items():
            d = {j: np.array([arbol.DEM[i, j, n] for n in ns]) for j in J}
            m = {j: (pi * d[j]).sum() for j in J}
            sd = {j: np.sqrt((pi * (d[j] - m[j]) ** 2).sum()) for j in J}
            cor = [(pi * (d[a] - m[a]) * (d[b] - m[b])).sum() / (sd[a] * sd[b])
                   for x, a in enumerate(J) for b in J[x + 1:]]
            tot = sum(d[j] for j in J)
            mt = (pi * tot).sum()
            filas.append(dict(periodo=p, vino=i, cv_etiquetas=np.mean([sd[j] / m[j] for j in J]),
                              correlacion_media=np.mean(cor),
                              cv_total_vino=np.sqrt((pi * (tot - mt) ** 2).sum()) / mt))
    return pd.DataFrame(filas)


ESTRUCTURAS = {"Ambos +": {1: "+", 2: "+"}, "Ambos −": {1: "-", 2: "-"},
               "V1 −, V2 +": {1: "-", 2: "+"}, "V1 +, V2 −": {1: "+", 2: "-"}}


def sens_correlacion(cvs=(0.1, 0.3, 0.5), PLs=POLITICAS, estructuras=ESTRUCTURAS, **kw):
    """Costo de cada politica para cada estructura de correlacion y CV. Devuelve (resultados, estadisticas)."""
    filas, est = [], []
    for cv in cvs:
        for nombre, signo in estructuras.items():
            arbol = arbol_correlacion(cv, signo)
            e = estadisticas_arbol(arbol).assign(cv=cv, estructura=nombre)
            est.append(e)
            for PL in PLs:
                r = modelo.resolver(PL=PL, datos_mod=arbol, **kw)
                filas.append(dict(cv=cv, estructura=nombre, PL=PL, politica=modelo._nombre_politica(PL),
                                  estado=r["status"], costo=r["obj"], **r["costo_detalle"], **r["kpis"],
                                  postergables=" ".join(f"V{i}E{j}" for i, j in r["postergables"]) or "-",
                                  tiempo_resolucion_s=r["tamano"]["Tiempo resolución (s)"]))
    return pd.DataFrame(filas), pd.concat(est, ignore_index=True)


def graficar_correlacion(df, estructuras=ESTRUCTURAS):
    """Una columna por CV. Arriba: costo esperado; abajo: ahorro frente a A stock. Eje x: estructura de correlacion."""
    cvs = sorted(df["cv"].unique())
    nombres = list(estructuras)
    x = np.arange(len(nombres))
    fig, ax = plt.subplots(2, len(cvs), figsize=(6 * len(cvs), 9), squeeze=False)
    for c, cv in enumerate(cvs):
        t = df[df["cv"] == cv]
        for f, (col, pols) in enumerate([("costo", POLITICAS), ("ahorro", POLITICAS[1:])]):
            a = ax[f, c]
            w = 0.8 / len(pols)
            for k, PL in enumerate(pols):
                v = []
                for n in nombres:
                    fila = t[(t["estructura"] == n) & (t["PL"] == PL)]["costo"].iloc[0]
                    ref = t[(t["estructura"] == n) & (t["PL"] == 0)]["costo"].iloc[0]
                    v.append(fila if col == "costo" else 100 * (1 - fila / ref))
                nom = modelo._nombre_politica(PL)
                a.bar(x + (k - (len(pols) - 1) / 2) * w, v, width=w, color=modelo._color_pol(nom), label=nom,
                      edgecolor=modelo.SUPERFICIE, linewidth=1)
            a.set_xticks(x, [n.replace(", ", ",\n") for n in nombres], fontsize=9)
            a.grid(axis="x", visible=False)
            if col == "costo":
                a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "$" + modelo._corto(v)))
                a.set_title(f"CV = {modelo._num(100 * cv)} %\nCosto esperado", linespacing=1.5)
            else:
                a.set_ylim(0, 105)
                a.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f} %"))
                a.set_title("Ahorro frente a A stock")
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.01, 0.955), ncol=len(l))
    fig.suptitle("Sensibilidad a la correlación entre las etiquetas de cada vino", x=0.01, ha="left",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def _etiqueta_cv(n):
    return "Árbol base\n(CV 0,26–0,61)" if n == -1 else f"CV = {modelo._num(100 * n)} %"


# ---------------------------------------------------------------------------
# Todo junto
# ---------------------------------------------------------------------------
def sens_atraso_capacidad(costos=(1500, 5000, 15000), capacidades=(84, 63, 42), PLs=POLITICAS):
    """Costo de atraso x capacidad de la linea: con 84 h nunca hay atrasos, por lo que solo con
    capacidad ajustada el costo de atraso puede afectar la decision."""
    out = []
    for T in capacidades:
        d = _corridas("Cb_bl", costos, PLs, T_prod=T)
        d["T_prod"] = T
        out.append(d)
    return pd.concat(out, ignore_index=True)


def cotas_por_atraso(costos=(1500, 5000, 15000), PLs=(0, 5)):
    """VSS y EVPI para cada costo de atraso (el VSS depende fuertemente de el)."""
    filas = []
    for c in costos:
        for PL in PLs:
            r = cotas(PL, Cb_bl=c)[0]
            filas.append({"Cb_bl": c, **r})
    return pd.DataFrame(filas)


def _guardar(fig, nombre):
    modelo.exportar_figura(fig, nombre, carpeta=SALIDA, dpi=300)   # PNG 300 dpi + SVG vectorial (para PPT)
    plt.close(fig)


def _csv(df, nombre, index=True):
    df.to_csv(os.path.join(SALIDA, nombre + ".csv"), index=index, encoding="utf-8-sig")


def correr_todo():
    os.makedirs(SALIDA, exist_ok=True)

    tab, ws, plan = tabla_cotas((0, 5))
    _csv(tab, "cotas_vss_evpi"); _csv(ws, "cotas_ws_trayectorias", index=False)
    _csv(plan, "cotas_plan_periodo1", index=False)
    _guardar(graficar_cotas(tab), "cotas_vss_evpi")
    print(tab.T.to_string())

    sa = sens_atraso()
    _csv(sa, "sens_costo_atraso", index=False)
    _guardar(graficar_sens(sa, "Cb_bl", lambda n: f"${modelo._num(n)}", "Costo de atraso ($ por botella y período)",
                           "Sensibilidad al costo de atraso", 15000), "sens_costo_atraso")

    sc = sens_atraso_capacidad()
    _csv(sc, "sens_costo_atraso_x_capacidad", index=False)
    for T in sorted(sc["T_prod"].unique(), reverse=True):
        _guardar(graficar_sens(sc[sc["T_prod"] == T], "Cb_bl", lambda n: f"${modelo._num(n)}",
                               "Costo de atraso ($ por botella y período)",
                               f"Sensibilidad al costo de atraso con {T} h de línea", 15000),
                 f"sens_costo_atraso_{T}h")
    print(sc.pivot_table(index=["T_prod", "Cb_bl"], columns="PL", values="costo").round(0).to_string())
    ca = cotas_por_atraso()
    _csv(ca, "cotas_por_costo_atraso", index=False)
    print(ca[["Cb_bl", "PL", "RP", "EEV", "WS", "VSS", "EVPI"]].round(0).to_string())

    sv = sens_variabilidad()
    _csv(sv, "sens_variabilidad", index=False)
    _csv(cv_por_producto_periodo(), "cv_arbol_base", index=False)
    _guardar(graficar_sens(sv, "cv", _etiqueta_cv, "Variabilidad de la demanda (desviación estándar / media)",
                           "Sensibilidad a la variabilidad de la demanda", None), "sens_variabilidad")
    print(sv.pivot(index="cv", columns="politica", values="costo").round(0).to_string())

    sk, ek = sens_correlacion()
    _csv(sk, "sens_correlacion", index=False)
    _csv(ek, "sens_correlacion_estadisticas_logradas", index=False)
    _guardar(graficar_correlacion(sk), "sens_correlacion")

    sr = sens_alfa()
    _csv(sr, "sens_valor_residual", index=False)
    _guardar(graficar_sens(sr, "alfa", lambda n: f"α = {modelo._num(n, 1)}", "Valor residual (α, fracción del costo de mantención)",
                           "Sensibilidad al valor residual del inventario final", 0), "sens_valor_residual")
    pl = plan_inicial_igual()
    _csv(pl, "valor_residual_verificacion_plan", index=False)
    print(sr.pivot(index="alfa", columns="politica", values="costo").round(0).to_string())
    print(pl.to_string())


if __name__ == "__main__":
    correr_todo()
