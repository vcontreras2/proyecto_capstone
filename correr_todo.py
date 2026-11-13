"""
Corre TODO una sola vez y congela los numeros.

    python correr_todo.py            # analisis.py + escalabilidad.py + horizonte.py + numeros congelados (~30 min)
    python correr_todo.py --forzar   # vuelve a correr aunque ya exista una corrida congelada

Deja en resultados/:
  * todos los CSV y figuras de los tres scripts,
  * numeros_congelados.csv: los numeros que se citan en el informe y el PPT (ver congelar.py),
  * ejecucion.json: fecha, duracion de cada etapa, semilla, limites de tiempo, versiones, equipo y huella (SHA-256)
    de cada CSV, para comprobar que dos personas tienen exactamente los mismos resultados.

Si resultados/ejecucion.json ya existe el script no corre: asi se evita generar resultados distintos sin querer.
"""
import datetime
import hashlib
import importlib.metadata as md
import inspect
import json
import os
import platform
import sys
import time

import analisis
import congelar
import escalabilidad
import horizonte
import modelo

EJEC = os.path.join("resultados", "ejecucion.json")


def _huellas(carpeta="resultados"):
    """SHA-256 de cada CSV de resultados (las figuras llevan metadatos de fecha, por eso se omiten)."""
    out = {}
    for base, _, archivos in os.walk(carpeta):
        for a in sorted(archivos):
            if not a.endswith(".csv") or a.endswith("_avance.csv"):
                continue
            ruta = os.path.join(base, a)
            with open(ruta, "rb") as fh:
                out[os.path.relpath(ruta, carpeta).replace(os.sep, "/")] = hashlib.sha256(fh.read()).hexdigest()
    return out


def main(forzar=False):
    if os.path.exists(EJEC) and not forzar:
        print(f"Ya existe {EJEC}: los resultados estan congelados. Use --forzar si realmente quiere volver a correr.")
        return 1
    inicio = datetime.datetime.now()
    etapas = {}
    for nombre, fn in (("analisis", analisis.correr_todo), ("escalabilidad", escalabilidad.correr_todo),
                       ("horizonte", horizonte.correr_todo)):
        t0 = time.perf_counter()
        print(f"=== {nombre} ===", flush=True)
        fn()
        etapas[nombre] = round(time.perf_counter() - t0, 1)
    n = len(congelar.generar())
    fin = datetime.datetime.now()
    pesc, phor = inspect.signature(escalabilidad.correr).parameters, inspect.signature(horizonte.correr).parameters
    equipo = {}
    try:
        equipo = modelo.info_equipo().iloc[:, 0].astype(str).to_dict()
    except Exception:
        pass
    info = dict(
        inicio=inicio.isoformat(timespec="seconds"), fin=fin.isoformat(timespec="seconds"),
        duracion_total_s=round((fin - inicio).total_seconds(), 1), duracion_por_etapa_s=etapas,
        semilla_solver=modelo.SEMILLA,
        escalabilidad=dict(limite_tiempo_s=pesc["tiempo_max"].default, politicas=list(pesc["PLs"].default)),
        horizonte=dict(limite_tiempo_por_ventana_s=phor["tiempo_max"].default, ventanas=list(phor["ventanas"].default)),
        numeros_congelados=n,
        versiones={p: md.version(p) for p in ("pulp", "highspy", "pandas", "numpy", "matplotlib", "networkx")},
        python=platform.python_version(), equipo=equipo,
        sha256_csv=_huellas(),
        nota=("Las corridas que terminan en 'Time limit' dependen de la velocidad del computador: con otra maquina, "
              "la mejor solucion hallada puede ser distinta. Por eso se usan estos numeros y no se vuelve a correr."))
    with open(EJEC, "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2, ensure_ascii=False)
    print(f"Listo en {info['duracion_total_s'] / 60:.1f} min. {n} numeros congelados. Detalle en {EJEC}")
    return 0


if __name__ == "__main__":
    sys.exit(main(forzar="--forzar" in sys.argv))
