"""
Numeros congelados: arma resultados/numeros_congelados.csv a partir de los CSV de una corrida.

La idea es correr los programas UNA vez (correr_todo.py) y que el informe, el PPT y el resto del equipo usen solo
estos numeros, para que nadie trabaje con resultados distintos. Cada fila trae su valor, su unidad y el archivo
de donde sale. Los porcentajes estan en escala 0-100.

Uso:   python congelar.py        (lo llama correr_todo.py; se puede correr solo si ya existen los CSV)
"""
import os

import pandas as pd

R = "resultados"
PLS = (0, 2, 3, 4, 5)


def _leer(ruta, nombre):
    return pd.read_csv(os.path.join(ruta, nombre))


def generar(ruta=R):
    filas = []

    def f(seccion, concepto, valor, unidad, fuente):
        filas.append(dict(seccion=seccion, concepto=concepto, valor=valor, unidad=unidad, fuente=fuente))

    def costo(df, **cond):
        q = df
        for k, v in cond.items():
            q = q[q[k] == v]
        return float(q["costo"].iloc[0])

    # --- caso base y tamano del modelo
    vr = _leer(ruta, "sens_valor_residual.csv")
    for PL in PLS:
        c = costo(vr, alfa=0, PL=PL)
        f("Caso base", f"Costo esperado PL={PL}", c, "$", "sens_valor_residual.csv")
        if PL:
            f("Caso base", f"Ahorro PL={PL} vs A stock", 100 * (1 - c / costo(vr, alfa=0, PL=0)), "%", "sens_valor_residual.csv")
    esc = _leer(ruta, "escalabilidad/escalabilidad.csv")
    b = esc[(esc.vinos == 2) & (esc.periodos == 3) & (esc.PL == 5)].iloc[0]
    f("Tamano del modelo base", "Variables", b["variables"], "n", "escalabilidad/escalabilidad.csv")
    f("Tamano del modelo base", "Variables enteras o binarias", b["binarias_enteras"], "n", "escalabilidad/escalabilidad.csv")
    f("Tamano del modelo base", "Restricciones (PuLP; 331 sin contar las que fijan valores)", b["restricciones"], "n",
      "escalabilidad/escalabilidad.csv")

    # --- valor residual
    for a in (0.5, 0.9):
        for PL in (0, 5):
            f("Valor residual", f"Costo esperado alfa={a} PL={PL}", costo(vr, alfa=a, PL=PL), "$", "sens_valor_residual.csv")
        f("Valor residual", f"Ahorro PL=5 vs A stock alfa={a}",
          100 * (1 - costo(vr, alfa=a, PL=5) / costo(vr, alfa=a, PL=0)), "%", "sens_valor_residual.csv")

    # --- VSS y EVPI
    cot = _leer(ruta, "cotas_vss_evpi.csv").set_index("PL")
    for PL in cot.index:
        for col, u in (("EV", "$"), ("WS", "$"), ("RP", "$"), ("EEV", "$"), ("VSS", "$"), ("EVPI", "$")):
            f("VSS y EVPI", f"{col} PL={PL}", cot.loc[PL, col], u, "cotas_vss_evpi.csv")
        f("VSS y EVPI", f"VSS/EEV PL={PL}", 100 * cot.loc[PL, "VSS_pct"], "%", "cotas_vss_evpi.csv")
        f("VSS y EVPI", f"EVPI/RP PL={PL}", 100 * cot.loc[PL, "EVPI_pct"], "%", "cotas_vss_evpi.csv")
    ca = _leer(ruta, "cotas_por_costo_atraso.csv")
    for _, r in ca.iterrows():
        f("VSS y EVPI", f"VSS PL={int(r.PL)} costo de atraso={int(r.Cb_bl)}", r["VSS"], "$", "cotas_por_costo_atraso.csv")

    # --- costo de atraso x capacidad
    xc = _leer(ruta, "sens_costo_atraso_x_capacidad.csv")
    for T in (84, 63, 42):
        for cb in (1500, 5000, 15000):
            c0, c5 = costo(xc, T_prod=T, Cb_bl=cb, PL=0), costo(xc, T_prod=T, Cb_bl=cb, PL=5)
            tag = f"{T} h, atraso {cb}"
            f("Costo de atraso", f"Costo A stock ({tag})", c0, "$", "sens_costo_atraso_x_capacidad.csv")
            f("Costo de atraso", f"Costo PL=5 ({tag})", c5, "$", "sens_costo_atraso_x_capacidad.csv")
            f("Costo de atraso", f"Ahorro PL=5 vs A stock ({tag})", 100 * (1 - c5 / c0), "%", "sens_costo_atraso_x_capacidad.csv")

    # --- variabilidad
    sv = _leer(ruta, "sens_variabilidad.csv")
    for cv in sorted(sv.cv.unique()):
        nom = "arbol base" if cv < 0 else f"CV {int(round(cv * 100))} %"
        c0 = costo(sv, cv=cv, PL=0)
        f("Variabilidad", f"Costo A stock ({nom})", c0, "$", "sens_variabilidad.csv")
        for PL in (2, 5):
            f("Variabilidad", f"Ahorro PL={PL} vs A stock ({nom})", 100 * (1 - costo(sv, cv=cv, PL=PL) / c0), "%",
              "sens_variabilidad.csv")

    # --- correlacion
    sk = _leer(ruta, "sens_correlacion.csv")
    for cv in sorted(sk.cv.unique()):
        for est in sk.estructura.unique():
            q = sk[(sk.cv == cv) & (sk.estructura == est)]
            c0 = costo(q, PL=0)
            f("Correlacion", f"Costo A stock (CV {int(round(cv * 100))} %, {est})", c0, "$", "sens_correlacion.csv")
            for PL in (2, 3, 4, 5):
                f("Correlacion", f"Ahorro PL={PL} vs A stock (CV {int(round(cv * 100))} %, {est})",
                  100 * (1 - costo(q, PL=PL) / c0), "%", "sens_correlacion.csv")

    # --- escalabilidad
    for (v, p), g in esc.groupby(["vinos", "periodos"]):
        tag = f"{v} vinos, {p} periodos"
        a, h = g[g.PL == 0].iloc[0], g[g.PL == g.productos.iloc[0]].iloc[0]   # PL = todos los productos
        f("Escalabilidad", f"Costo A stock ({tag})", a["costo"], "$", "escalabilidad/escalabilidad.csv")
        f("Escalabilidad", f"Tiempo A stock ({tag})", a["t_resolucion_s"], "s", "escalabilidad/escalabilidad.csv")
        f("Escalabilidad", f"Costo postergar todo ({tag}; estado {h['estado']})", h["costo"], "$", "escalabilidad/escalabilidad.csv")
        f("Escalabilidad", f"Tiempo postergar todo ({tag}; estado {h['estado']})", h["t_resolucion_s"], "s",
          "escalabilidad/escalabilidad.csv")
        f("Escalabilidad", f"Brecha postergar todo ({tag})", 100 * h["brecha"], "%", "escalabilidad/escalabilidad.csv")
        f("Escalabilidad", f"Ahorro postergar todo vs A stock ({tag}; estado {h['estado']})", 100 * h["ahorro_vs_a_stock"], "%",
          "escalabilidad/escalabilidad.csv")

    # --- horizonte rodante
    hv = _leer(ruta, "horizonte/horizonte_vs_completo.csv")
    for _, r in hv[hv.politica == "todos"].iterrows():
        tag = f"{int(r.vinos)} vinos, {int(r.periodos)} periodos, ventana {int(r.ventana)}"
        f("Horizonte rodante", f"Costo postergar todo ({tag})", r["costo"], "$", "horizonte/horizonte_vs_completo.csv")
        f("Horizonte rodante", f"Tiempo total ({tag})", r["t_total_s"], "s", "horizonte/horizonte_vs_completo.csv")
        f("Horizonte rodante", f"Maximo sobrecosto vs optimo ({tag})", 100 * r["max_sobrecosto_vs_optimo"], "%",
          "horizonte/horizonte_vs_completo.csv")
    hl = _leer(ruta, "horizonte/horizonte_periodos_largos.csv")
    for _, r in hl.iterrows():
        tag = f"{int(r.vinos)} vinos, {int(r.periodos)} periodos, ventana {int(r.ventana)}, {r.politica}"
        f("Horizonte largo", f"Costo ({tag})", r["costo"], "$", "horizonte/horizonte_periodos_largos.csv")
        f("Horizonte largo", f"Tiempo total ({tag})", r["t_total_s"], "s", "horizonte/horizonte_periodos_largos.csv")

    df = pd.DataFrame(filas)
    df.to_csv(os.path.join(ruta, "numeros_congelados.csv"), index=False, encoding="utf-8-sig")
    return df


if __name__ == "__main__":
    d = generar()
    print(len(d), "numeros escritos en", os.path.join(R, "numeros_congelados.csv"))
