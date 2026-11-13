# Modelo de embotellado y etiquetado con postergación

Modelo de optimización estocástica (MILP multietapa sobre un árbol de escenarios) que planifica la línea de **embotellado y etiquetado de vinos**, comparando tres políticas de producción:

| Política | Descripción | Cómo se obtiene |
|---|---|---|
| **A stock** | Todo se embotella y etiqueta de una vez; no se posterga nada. | `resolver(PL=0)` |
| **Híbrida** | Hasta `PL` productos (vino, etiqueta) pueden embotellarse *sin etiqueta* y etiquetarse después, cuando se conoce la demanda. | `resolver(PL=2)`, `PL=3`, … |
| **Contra pedido** | No se permite inventario de producto terminado (R15: `sbl = 0`). | `resolver(contra_pedido=True)` |

La formulación está en [formulacion_compacta.pdf](formulacion_compacta.pdf) y los datos de la instancia en [Anexo a Datos.pdf](Anexo%20a%20Datos.pdf). El modelo se resuelve con **PuLP + HiGHS**, ambos gratis (no hace falta Gurobi).

---

## Estructura del proyecto

```
.
├── modelo.py               # Construcción, resolución y gráficos del modelo
├── datos.py                # Instancia: árbol de escenarios, probabilidades, demanda y parámetros
├── analisis.py             # Entrega 3: VSS/EVPI, sensibilidades (atraso, valor residual, variabilidad, correlación)
├── escalabilidad.py        # Entrega 3: más vinos y más períodos (resultados y tiempos)
├── horizonte.py            # Entrega 3: política de horizonte rodante
├── correr_todo.py          # Entrega 3: corre los tres scripts anteriores UNA vez y congela los números
├── congelar.py             # Entrega 3: arma resultados/numeros_congelados.csv (lo llama correr_todo.py)
├── visualizacion.ipynb     # Notebook para correr el modelo y ver los resultados
├── resultados/             # CSV y figuras generados por los scripts (ver "Guía de los archivos de resultados")
│   ├── numeros_congelados.csv  # los números oficiales para el informe y el PPT
│   ├── ejecucion.json      #   registro de la corrida: fecha, semilla, versiones, equipo y huellas de los CSV
│   ├── svg/                #   las mismas figuras en vectorial
│   ├── escalabilidad/      #   resultados de escalabilidad.py
│   └── horizonte/          #   resultados de horizonte.py
├── formulacion_compacta.pdf
├── formulacion_E3.pdf      # Formulación de la Entrega 3 (agrega el valor residual)
└── Anexo a Datos.pdf
```

---

## ¿Qué hace `modelo.py`?

### 1. Datos de entrada (`datos.py`)

- **Árbol de escenarios** (`PADRE`): 11 nodos en 3 periodos. El nodo 1 es la raíz y los nodos 5–11 son hojas (último periodo).
- **Probabilidades** (`PROB`): la probabilidad de cada nodo es el producto de las probabilidades condicionales desde la raíz.
- **Productos**: 2 vinos; el vino 1 tiene 3 etiquetas y el vino 2 tiene 2, así que hay 5 productos `(vino, etiqueta)`.
- **Demanda** (`DEM`): botellas por producto y nodo.
- **Parámetros** (`PARAMS`, Tabla 2): horas de línea, estanques, tiempos y costos de set-up, costos de inventario y de atraso, cotas big-M.

### 2. Variables de decisión (por nodo `n`)

| Variable | Significado |
|---|---|
| `wbl[i,j,n]` | Botellas del vino `i` embotelladas **y etiquetadas** con la etiqueta `j` |
| `wb[i,n]` | Botellas del vino `i` embotelladas **sin etiqueta** |
| `wl[i,j,n]` | Botellas sin etiqueta del inventario que se **etiquetan** con `j` |
| `sb[i,n]` / `sbl[i,j,n]` | Inventario sin etiqueta / de producto terminado |
| `bbl[i,j,n]` | Demanda atrasada (backlog) |
| `y[i,n]`, `u[i,n]` | Estanques usados (entero) y fracción sin usar del último estanque |
| `zb`, `zbl`, `zl` | Set-ups binarios de cada operación |
| `zpos[i,j]` | 1 si el producto `(i,j)` se elige como **postergable** |

### 3. Función objetivo

Minimiza el **costo esperado**, ponderando cada nodo por su probabilidad `π_n`:

> costo de set-ups + costo de inventario (con y sin etiqueta) + costo de atrasos − valor residual del inventario final

**Valor residual (Entrega 3).** El inventario que queda al final del último período se valora en `alfa × Cs` por botella (`Cs_b` sin etiqueta, `Cs_bl` terminada). Solo cuenta en los nodos hoja y se resta de la función objetivo. `alfa` está en `PARAMS` (por defecto 0, que reproduce la Entrega 2) y debe cumplir `0 ≤ alfa < 1`: con `alfa ≥ 1` el modelo sobreproduciría para acumular valor residual, y `resolver` lanza un error. El valor residual se reporta como componente aparte (`"Valor residual"` en `costo_detalle`, `valor_residual` en `costo`), para que el costo de inventario siga siendo comparable con la Entrega 2.

### 4. Restricciones principales

| Restricción | Descripción |
|---|---|
| R1 | Máximo `K_tank` estanques por periodo |
| R2 | Fracción sin usar de estanque ≤ `U` |
| R3 | Balance de litros: estanques usados = botellas embotelladas × `V_bot` |
| R4 | Balance de inventario sin etiqueta |
| R5 | Balance de producto terminado con atrasos |
| R6 | Condiciones iniciales mediante un nodo ficticio (`AUX = 0`) |
| R7 | Horas de línea por periodo ≤ `T_prod` |
| R8–R10 | Activación de set-ups (big-M) |
| R11 | Solo se etiquetan después los productos marcados como postergables |
| R12 | A lo más `PL` productos postergables |
| R13 | No se produce para inventario en el último periodo |
| R15 | (Solo contra pedido) sin inventario terminado |

### 5. Funciones disponibles

| Función | Qué hace |
|---|---|
| `resolver(PL=2, contra_pedido=False, datos_mod=datos, tiempo_max=300, verbose=False, fijar=None, estado_inicial=None, semilla=None, **override)` | Construye y resuelve el modelo y devuelve un `dict` con el estado, el costo, su desglose, DataFrames (`nodos`, `productos`, `vinos`) y los productos postergables elegidos. `override` permite cambiar parámetros, p. ej. `Cb_bl=1000` o `alfa=0.5`. `fijar` fija decisiones del nodo raíz (lo usa el EEV) y `estado_inicial` parte desde un inventario/atraso distinto de cero (lo usa el horizonte rodante). |
| `exportar_figura(fig, nombre, carpeta="figuras", dpi=300)` | Guarda una figura como PNG a 300 dpi y como SVG vectorial, para pegarla en un PPT sin perder calidad (ver más abajo). |
| `resumen(res)` | Imprime la política, el estado, el costo esperado y su desglose. |
| `comparar_politicas(PLs=(0,2,3,4,5))` | Resuelve todas las políticas y devuelve una tabla comparativa. |
| `sensibilidad(parametro, valores, PLs)` | Resuelve cada política para varios valores de un parámetro (por defecto `T_prod`). |
| `graficar_demanda()` | Demanda por nodo, apilada por producto. |
| `graficar_arbol(res)` | Dibuja el árbol de escenarios, coloreado según el % de horas de línea usadas. |
| `graficar_decisiones(res)` | Muestra producción, inventario, atrasos y horas por nodo (4 gráficos). |
| `graficar_comparacion(tabla)` | Desglose del costo por política y costo total en escala logarítmica. |
| `graficar_sensibilidad(sens, base)` | Costo de cada política según el parámetro variado. |
| `tabla_plan_inicial(res)`, `tabla_comparacion(tabla)`, `tabla_parametros(params)` | Tablas con formato listas para presentar. |
| `tabla_costos(tabla)`, `graficar_costos_detalle(tabla)` | Los 6 términos de la función objetivo por política: 3 tipos de set-up, inventario sin etiqueta, inventario terminado y atrasos. |
| `tabla_kpis(tabla)`, `graficar_kpis(tabla)` | KPIs operativos por política: nivel de servicio, utilización, set-ups e inventario promedio. |
| `sensibilidad(..., detalle=True)`, `graficar_sensibilidad_kpis(det)` | Costos y KPIs para cada nivel del parámetro y cada política. |

| `graficar_politicas(tabla)` | Costo esperado de contra pedido, a stock e híbrida (escala log). |
| `graficar_base_vs_alternativa(tabla)`, `tabla_base_vs_alternativa(tabla)` | KPIs y costos del caso base (a stock) vs. la híbrida PL=5. |
| `graficar_puente_costos(tabla)` | Gráfico de cascada: de dónde viene el ahorro, componente por componente. |
| `graficar_inventario_nodos(res_base, res_alt)` | Inventario al cierre de cada nodo en ambas políticas. |
| `tabla_tiempos(tabla)`, `info_equipo()` | Tamaño del modelo, tiempos de resolución y datos del equipo. |

`resolver()` devuelve además `res["costo_detalle"]` (los 6 costos más el valor residual), `res["kpis"]` y `res["tamano"]` (variables, restricciones, tiempos y `"Brecha MIP"`). Los KPIs son valores esperados, ponderados por la probabilidad de cada nodo.

> **Estado del solver.** `res["status"]` es `"Optimal"` solo si HiGHS demostró el óptimo. Si se detuvo por `tiempo_max`, vale `"Time limit"` y el costo es el de la mejor solución encontrada (una cota superior del óptimo); la brecha está en `res["tamano"]["Brecha MIP"]`. (PuLP por sí solo informaría "Optimal" en ese caso, por eso `resolver` lo corrige.)

Todos los gráficos usan el mismo estilo (`aplicar_estilo()`), una paleta apta para daltonismo y números en formato chileno (1.234.567,8). Para exportar una figura en alta resolución: `fig = modelo.graficar_arbol(res); fig.savefig("arbol.png")`.

---

## Cómo correrlo localmente

### Requisitos

- Python 3.11 o superior (probado con 3.13.3; pandas 3 y numpy 2.5 no funcionan con versiones anteriores)

### 1. Crear un entorno virtual (opcional, pero recomendado)

```bash
python -m venv .venv
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

[requirements.txt](requirements.txt) fija las versiones con las que se probó el proyecto:

| Paquete | Versión | Para qué |
|---|---|---|
| `pulp` | 3.3.0 | Modelado del MILP |
| `highspy` | 1.15.1 | Solver HiGHS (lo usa PuLP con `pulp.HiGHS`) |
| `pandas` | 3.0.5 | Tablas de resultados |
| `numpy` | 2.5.2 | Cálculo numérico |
| `matplotlib` | 3.11.2 | Gráficos |
| `networkx` | 3.7 | Árbol de escenarios |
| `jupyter` | 1.1.1 | Notebook de visualización |

### 3. Ejecutar

**Opción A: desde la terminal**

```bash
python -c "import modelo; r = modelo.resolver(PL=2); modelo.resumen(r)"
```

Salida esperada:

```
Politica: hibrida PL=2 | estado: Optimal | costo esperado: $1,071,131
  set-ups $13,100 | inventario $1,058,031 | atrasos $0
  productos postergables elegidos (vino, etiqueta): [(1, 1), (1, 2)]
```

**Opción B: script de Python**

```python
import matplotlib.pyplot as plt
import modelo

# Una política
res = modelo.resolver(PL=2)
modelo.resumen(res)
print(res["productos"].head())

modelo.graficar_arbol(res)
modelo.graficar_decisiones(res)

# Comparar todas las políticas
tabla = modelo.comparar_politicas()
print(tabla)
modelo.graficar_comparacion(tabla)

plt.show()
```

**Opción C: notebook**

```bash
jupyter notebook visualizacion.ipynb
```

> Ejecuta los comandos desde la carpeta del proyecto, porque `modelo.py` importa `datos.py` de forma relativa.

### Análisis de sensibilidad

Puedes cambiar cualquier parámetro de `PARAMS` sin editar `datos.py`:

```python
res = modelo.resolver(PL=3, Cb_bl=1000, T_prod=100)
tabla = modelo.comparar_politicas(T_prod=70)
```

Para usar otra instancia, crea un módulo con la misma estructura que `datos.py` (`PADRE`, `PROB`, `DEM`, `ETIQUETAS`, `PARAMS`) y pásalo con `resolver(datos_mod=mi_modulo)`.

---

## Análisis de la Entrega 3

### Números congelados: correr una sola vez

Los resultados de la Entrega 3 se generaron **una sola vez** con `python correr_todo.py` (unos 30 min), y esos son los números que usamos todos en el informe y el PPT. **No hace falta, y no conviene, volver a correr los programas**: si se repite, los casos que terminan en `Time limit` pueden dar otra brecha según la velocidad del computador.

- `resultados/numeros_congelados.csv` reúne las cifras que se citan (caso base, valor residual, VSS/EVPI, costo de atraso, variabilidad, correlación, escalabilidad y horizonte rodante). Cada fila trae `seccion`, `concepto`, `valor`, `unidad` (`$`, `%` en escala 0–100, `s`, `n`) y el archivo de origen (`fuente`). Es la referencia para citar cualquier número.
- `resultados/ejecucion.json` registra cuándo se corrió, cuánto tardó cada etapa, la semilla, los límites de tiempo, las versiones de Python y de las librerías, el equipo, y la huella SHA-256 de cada CSV. Para comprobar que dos personas tienen los mismos resultados, basta comparar esas huellas.
- `correr_todo.py` **se niega a correr si `ejecucion.json` ya existe**, para que nadie regenere resultados distintos por accidente. Solo con `python correr_todo.py --forzar` vuelve a ejecutar, y sobrescribe `resultados/`.

**Semilla.** La única fuente de aleatoriedad del proyecto es el solver: los datos y los árboles son deterministas. La semilla de HiGHS está fija en `modelo.SEMILLA = 0` y se pasa en cada resolución (`resolver(..., semilla=...)` permite cambiarla). Con la misma semilla y el mismo computador, un mismo problema da la misma solución. En los problemas difíciles que se detienen por límite de tiempo la semilla cambia mucho el resultado (en una prueba, la semilla 0 dio $4,6 MM y la semilla 7 dio $1,13 MM para el mismo caso a 30 s), y el límite de tiempo es de reloj, así que otro computador puede llegar a otra solución. Por eso se congelaron los números.

### Scripts por separado

Tres scripts generan todos los resultados de la Entrega 3 (`correr_todo.py` los ejecuta en secuencia). Se ejecutan **desde la carpeta del proyecto** y escriben en `resultados/`:

```bash
python analisis.py         # VSS/EVPI, costo de atraso, valor residual, variabilidad y correlación (unos minutos)
python escalabilidad.py    # más vinos y más períodos; PL=0 y PL=todos, límite de 180 s por corrida (unos 15–20 min)
python horizonte.py        # política de horizonte rodante
```

| Script | Qué responde | Funciones principales |
|---|---|---|
| `analisis.py` | ¿Cuánto vale modelar la incertidumbre (VSS, EVPI)? ¿Cómo cambian los resultados con el costo de atraso, el valor residual, la variabilidad y la correlación de la demanda? | `cotas(PL)`, `tabla_cotas()`, `sens_atraso()`, `sens_atraso_capacidad()`, `sens_alfa()`, `sens_variabilidad()`, `sens_correlacion()`, `arbol_variabilidad(cv)`, `arbol_correlacion(cv, signo)` |
| `escalabilidad.py` | ¿Qué pasa con un 3er o 4to vino, o con 4 o 5 períodos, en resultados y en tiempos? | `construir_instancia(n_vinos, n_periodos)`, `correr()`, `graficar_tiempos()`, `graficar_resultados()` |
| `horizonte.py` | Si el horizonte completo no se alcanza a resolver, ¿cuánto se pierde resolviendo ventanas de pocos períodos? | `politica_rodante(inst, PL, ventana)`, `correr()` |

**Definiciones usadas**

- **RP**: costo esperado óptimo del modelo estocástico (el modelo de siempre).
- **EV**: modelo determinista con la demanda esperada de cada período.
- **EEV**: se fija la decisión del período 1 del EV (producción y productos postergables) y se reoptimiza el resto en los 11 nodos.
- **WS**: cada una de las 7 trayectorias se resuelve con información perfecta y se pondera por la probabilidad de su hoja.
- **VSS = EEV − RP** y **EVPI = RP − WS**.
- **CV** = desviación estándar / media de la demanda de un producto en un período (ponderada por probabilidad). En `sens_variabilidad` y `sens_correlacion` se reescala la demanda para que todos los productos tengan el CV pedido, sin cambiar medias, probabilidades ni el período 1.
- **Correlación `+`**: todas las etiquetas de un vino suben y bajan juntas (correlación +1). **Correlación `−`**: la demanda total del vino casi no varía y cambia la mezcla entre etiquetas (−1 con 2 etiquetas, −0,5 con 3, el máximo posible).
- **Instancias de escalabilidad** (sintéticas, no hay datos reales): el vino 3 replica al vino 1 con demanda × 0,9 y el vino 4 al vino 2 con × 1,1; los períodos nuevos extienden el árbol con una cadena de Markov alto/medio/bajo usando las transiciones del árbol base (11, 28 y 69 nodos para 3, 4 y 5 períodos); la capacidad crece en proporción a la demanda esperada.
- **Horizonte rodante**: en cada nodo se resuelve solo el subárbol de `ventana` períodos que parte ahí, partiendo del inventario, los atrasos y la producción en tránsito reales; los productos postergables se deciden una vez en el período 1. Con ventana ≥ períodos coincide con el modelo completo.

### Guía de los archivos de resultados

Todas las figuras existen como **PNG a 300 dpi** y como **SVG vectorial** (en la subcarpeta `svg/` de cada carpeta). Para un PPT, conviene insertar el SVG (*Insertar → Imágenes*, requiere PowerPoint 2019 o Microsoft 365), que no se pixela al agrandarlo; si no se ve bien, usar el PNG. Para exportar una figura propia con el mismo formato: `modelo.exportar_figura(fig, "nombre")`.

Convenciones comunes: `PL` es el máximo de productos postergables (`0` = A stock); `politica` es el nombre legible (`A stock`, `Híbrida PL=5`); los costos son costo esperado en $; `estado` es `Optimal` o `Time limit` (ver "Estado del solver").

#### Números congelados y registro de la corrida (`correr_todo.py`)

| Archivo | Qué contiene |
|---|---|
| `numeros_congelados.csv` | 245 cifras oficiales en formato largo: `seccion` (Caso base, Valor residual, VSS y EVPI, Costo de atraso, Variabilidad, Correlacion, Escalabilidad, Horizonte rodante, Horizonte largo, Tamano del modelo base), `concepto`, `valor`, `unidad` y `fuente` (el CSV del que sale). Para citar un número en el informe o el PPT, buscarlo aquí. En las filas de escalabilidad, el `estado` (`Optimal` o `Time limit`) va dentro del concepto. |
| `ejecucion.json` | Registro de la corrida única: `inicio`, `fin`, `duracion_total_s` y por etapa, `semilla_solver`, límites de tiempo, versiones de las librerías, Python, equipo y `sha256_csv` (huella de cada CSV). |

#### VSS y EVPI (`analisis.py`)

| Archivo | Qué contiene |
|---|---|
| `cotas_vss_evpi.csv` | Una fila por política (PL=0 y PL=5) con `EV`, `WS`, `RP`, `EEV`, `VSS`, `EVPI` y sus porcentajes (`VSS_pct` = VSS/EEV, `EVPI_pct` = EVPI/RP). `EEV_postergables_libres` es una variante de EEV que solo fija la producción del período 1 y deja libres los postergables. `postergables_RP` y `postergables_EV` son los productos postergados en cada solución. `orden_ok` comprueba que EV ≤ WS ≤ RP ≤ EEV. |
| `cotas_vss_evpi.png` / `.svg` | Barras de WS, RP, EEV y EV por política, en escala logarítmica (difieren en órdenes de magnitud), con el VSS y el EVPI en el título. |
| `cotas_ws_trayectorias.csv` | Una fila por trayectoria (7 hojas) y política: `camino` de nodos, `prob` de la hoja, `costo` con información perfecta y productos postergados. WS es el promedio de `costo` ponderado por `prob`. |
| `cotas_plan_periodo1.csv` | Decisión del período 1: botellas producidas por producto en la solución estocástica (`produccion_RP`) y en la determinista (`produccion_EV`), junto con la demanda del período 1. Muestra por qué el EEV es tan malo: el EV produce justo lo esperado. |
| `cotas_por_costo_atraso.csv` | Las mismas cotas para cada costo de atraso (1.500, 5.000 y 15.000). El VSS depende casi linealmente de ese costo. |

#### Sensibilidad al costo de atraso (`analisis.py`)

Con 84 h de línea no hay atrasos en ninguna política, por lo que el costo de atraso no cambia nada; por eso se cruzó también con la capacidad.

| Archivo | Qué contiene |
|---|---|
| `sens_costo_atraso.csv` | Costo de atraso 1.500 / 5.000 / 15.000 × PL (0, 2, 3, 4, 5), con 84 h. Incluye los 6 costos por componente, el valor residual, los KPIs, los productos postergados y el tiempo de resolución. |
| `sens_costo_atraso.png` / `.svg` | Costo esperado, ahorro frente a A stock y nivel de servicio por política. Es idéntico a `sens_costo_atraso_84h.png`. |
| `sens_costo_atraso_x_capacidad.csv` | Lo mismo para cada capacidad `T_prod` (84, 63 y 42 h); la columna `T_prod` indica el nivel. Aquí sí hay atrasos y el costo de atraso importa. |
| `sens_costo_atraso_84h.png`, `_63h.png`, `_42h.png` (y `.svg`) | Una figura por capacidad. Con 63 y 42 h el eje de costos es logarítmico. |

#### Valor residual (`analisis.py`)

| Archivo | Qué contiene |
|---|---|
| `sens_valor_residual.csv` | `alfa` = 0, 0,5 y 0,9 × PL (0, 2, 3, 4, 5). La columna `Valor residual` es negativa (se resta del costo); `costo` es el total. Con `alfa = 0` reproduce la Entrega 2 (1.918.302 y 617.164). |
| `sens_valor_residual.png` / `.svg` | Costo esperado, ahorro frente a A stock y nivel de servicio por política y nivel de `alfa`. |
| `valor_residual_verificacion_plan.csv` | Comprueba que `alfa` no cambia el plan: `max_dif_botellas` es la mayor diferencia de producción respecto de `alfa = 0` (del orden de 1e-10) y `mismos_postergables` indica si se postergan los mismos productos. |

#### Variabilidad de la demanda (`analisis.py`)

| Archivo | Qué contiene |
|---|---|
| `cv_arbol_base.csv` | CV y media de cada producto y período del árbol original (equivale a la Tabla B.1 del informe de la Entrega 2). |
| `sens_variabilidad.csv` | Árbol original (`cv = -1`) y árboles con CV 10, 30 y 50 % (`cv = 0.1`, `0.3`, `0.5`) × PL. |
| `sens_variabilidad.png` / `.svg` | Costo, ahorro frente a A stock y nivel de servicio por nivel de variabilidad. |

#### Correlación de la demanda (`analisis.py`)

| Archivo | Qué contiene |
|---|---|
| `sens_correlacion.csv` | 4 estructuras × 3 CV × PL. `estructura`: `Ambos +`, `Ambos −`, `V1 −, V2 +` y `V1 +, V2 −` (signo de la correlación entre las etiquetas de cada vino). |
| `sens_correlacion.png` / `.svg` | Una columna por CV; arriba el costo esperado y abajo el ahorro frente a A stock, por estructura y política. |
| `sens_correlacion_estadisticas_logradas.csv` | CV (`cv_etiquetas`), correlación media entre etiquetas (`correlacion_media`) y CV de la demanda total del vino (`cv_total_vino`) realmente logrados en cada árbol y período. Sirve para citar la correlación efectiva: −1 en el Vino 2 y −0,5 en el Vino 1 en las estructuras negativas. |

#### Escalabilidad (`escalabilidad.py`, carpeta `resultados/escalabilidad/`)

| Archivo | Qué contiene |
|---|---|
| `escalabilidad.csv` | Una fila por combinación de vinos (2, 3, 4), períodos (3, 4, 5) y política (`politica` = `0` o `todos`, es decir, todos los productos postergables; `PL` es el valor numérico). Columnas de tamaño (`nodos`, `hojas`, `variables`, `binarias_enteras`, `restricciones`), de capacidad usada (`T_prod`, `K_tank`), tiempos (`t_construccion_s`, `t_resolucion_s`), `estado`, `brecha` (brecha MIP; 0 si es `Optimal`), `costo`, `ahorro_vs_a_stock` y KPIs. **Si `estado` es `Time limit`, el costo es el de la mejor solución encontrada: el ahorro real es al menos el reportado y la cota inferior del costo es `costo × (1 − brecha)`.** |
| `tiempos.png` / `.svg` | Tiempo de resolución y número de variables contra nodos del árbol (escala log). Los puntos huecos son corridas que alcanzaron el límite de tiempo, con su brecha. |
| `resultados.png` / `.svg` | Ahorro de postergar frente a A stock y costo esperado por vinos y períodos. Las barras rayadas y los puntos huecos son corridas con límite de tiempo. |
| `escalabilidad_avance.csv` | Copia parcial guardada durante la corrida (sirve si se interrumpe). Ignorarla si existe `escalabilidad.csv`. |
| `log.txt` | Salida en pantalla de la corrida (una línea por resolución). |

#### Horizonte rodante (`horizonte.py`, carpeta `resultados/horizonte/`)

`horizonte.py` usa los resultados de `escalabilidad.py` como referencia, así que conviene correr ese script primero (si no existe `escalabilidad/escalabilidad.csv`, se omite la comparación y las figuras). Cada ventana tiene un límite de 60 s.

| Archivo | Qué contiene |
|---|---|
| `horizonte.csv` | Una fila por vinos (2, 3, 4) × períodos (4, 5) × ventana (2 y 3 períodos) × política (`0` = A stock, `todos` = todos los productos postergables). `costo` es el costo esperado de la política de horizonte rodante evaluada sobre todo el árbol, con sus componentes (`setups`, `inventario`, `atrasos`), `servicio` e `inventario_prom`. `ventanas` es el número de problemas resueltos (uno por nodo del árbol). Tiempos: `t_total_s` (con construcción y lectura de resultados), `t_resolucion_total_s` (solo solver) y `t_max_ventana_s` (la ventana más lenta). `ventanas_limite_tiempo` y `brecha_max` indican si alguna ventana se detuvo por el límite de 60 s (en esta corrida ninguna). `postergables` son los productos habilitados en el período 1. |
| `horizonte_vs_completo.csv` | `horizonte.csv` más las columnas del modelo completo (`costo_completo`, `estado_completo`, `brecha_completo`, `t_completo_s`). `dif_costo_vs_completo` = costo rodante / costo completo − 1: es **negativo cuando el modelo completo no alcanzó el óptimo**, porque su costo es solo la mejor solución que encontró. `cota_inferior_completo` = `costo_completo × (1 − brecha_completo)` es una cota inferior del verdadero óptimo, y `max_sobrecosto_vs_optimo` = costo rodante / cota inferior − 1 es el máximo que el horizonte rodante podría estar por sobre el óptimo. |
| `comparacion_costos.png` / `.svg` | Costo de la política híbrida (todos los productos postergables): modelo completo contra horizonte rodante con ventana 3 y 2, por vinos y períodos. Las barras rayadas son corridas del modelo completo que alcanzaron el límite de tiempo (con su brecha). |
| `comparacion_tiempos.png` / `.svg` | Tiempo total de resolución de la política híbrida, en escala log, del modelo completo y del horizonte rodante. |
| `horizonte_periodos_largos.csv` | Horizonte rodante con ventana 2 para 2 y 4 vinos y 6 y 7 períodos (168 y 407 nodos), tamaños que el modelo completo no resuelve. Mismas columnas que `horizonte.csv`. No hay referencia del modelo completo, solo sirve para ver cómo crece el tiempo. |
| `horizonte_avance.csv`, `log.txt` | Copia parcial guardada durante la corrida y salida en pantalla. Ignorables si existe `horizonte.csv`. |

**Cómo leerlo.** Con ventana ≥ períodos, el horizonte rodante es el modelo completo. En esta instancia sintética la ventana de 2 períodos da el mismo costo que la de 3 en todos los casos; esto se debe a que el embotellado tiene un rezago de un período, y puede no cumplirse con otros datos.
