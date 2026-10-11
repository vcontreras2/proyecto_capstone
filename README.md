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
├── visualizacion.ipynb     # Notebook para correr el modelo y ver los resultados
├── formulacion_compacta.pdf
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

> costo de set-ups + costo de inventario (con y sin etiqueta) + costo de atrasos

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
| `resolver(PL=2, contra_pedido=False, tiempo_max=300, verbose=False, **override)` | Construye y resuelve el modelo y devuelve un `dict` con el estado, el costo, su desglose, DataFrames (`nodos`, `productos`, `vinos`) y los productos postergables elegidos. `override` permite cambiar parámetros, p. ej. `Cb_bl=1000`. |
| `resumen(res)` | Imprime la política, el estado, el costo esperado y su desglose. |
| `comparar_politicas(PLs=(0,2,3,4,5))` | Resuelve todas las políticas y devuelve una tabla comparativa. |
| `graficar_arbol(res)` | Dibuja el árbol de escenarios, coloreado según el % de horas de línea usadas. |
| `graficar_decisiones(res)` | Muestra producción, inventario, atrasos y horas por nodo (4 gráficos). |
| `graficar_comparacion(tabla)` | Barras apiladas del costo por política. |

---

## Cómo correrlo localmente

### Requisitos

- Python 3.9 o superior (probado con 3.13)

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
pip install pulp highspy pandas matplotlib networkx jupyter
```

> `highspy` instala el solver HiGHS que usa PuLP (`pulp.HiGHS`).

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
