# -*- coding: utf-8 -*-
import csv
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Iterable, Tuple,Optional
import heapq

# =========================
# Estructuras simples
# =========================
@dataclass
class Worker:
    home: int   # nodo hogar v_i
    r: float    # radio de caminata r_i

class Graph:
    def __init__(self, directed: bool = True):
        self.directed = directed
        self.edges: Dict[int, Dict[int, float]] = defaultdict(dict)

    def add_edge(self, u: int, v: int, cost: float) -> None:
        self.edges[u][v] = cost
        if not self.directed:
            self.edges[v][u] = cost

    def neighbors(self, u: int) -> Iterable[Tuple[int, float]]:
        return self.edges.get(u, {}).items()

    @property
    def nodes(self) -> List[int]:
        ns = set(self.edges.keys())
        for adj in self.edges.values():
            ns.update(adj.keys())
        return sorted(ns)

    def __len__(self) -> int:
        return sum(len(adj) for adj in self.edges.values())
#Cuantos edges tiene el grafo

# =========================
# Lectura de CSV (simple)
# =========================
def read_graph_csv(path: str, directed: bool = True, skip_header: bool = False) -> Graph:
    """
    Espera filas: i,j,cij (enteros para nodos, float para costo).
    """
    g = Graph(directed=directed)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        if skip_header:
            next(reader, None)
        for i_str, j_str, c_str in reader:
            g.add_edge(int(i_str), int(j_str), float(c_str))
    return g


def read_workers_csv(path: str, skip_header: bool = False) -> List[Worker]:
    """
    Espera filas: v_i,r_i (int, float).
    """
    workers: List[Worker] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        if skip_header:
            next(reader, None)
        for v_str, r_str in reader:
            workers.append(Worker(home=int(v_str), r=float(r_str)))
    return workers


def index_workers_by_node(workers: List[Worker]) -> Dict[int, List[int]]:
    idx = defaultdict(list)
    for k, w in enumerate(workers):
        idx[w.home].append(k)
    return dict(idx)
# te dice que trabajadores viven en cada nodo

# =========================
# Ejemplo mínimo de uso
# =========================
"""

if __name__ == "__main__":
    G_PATH = "dataset/dataset_enviar/grafo.csv"
    W_PATH = "dataset/dataset_enviar/instancia1.csv"

    g = read_graph_csv(G_PATH, directed=True, skip_header=False)
    workers = read_workers_csv(W_PATH, skip_header=False)
    idx = index_workers_by_node(workers)

    print(f"Nodos: {len(g.nodes)} | Aristas: {len(g)} | Trabajadores: {len(workers)}")
    if 0 in g.nodes:
        print("Vecinos de 0:", list(g.neighbors(0)))
"""



import heapq
from typing import Dict, Tuple, List

def dijkstra_verbose(g: Graph, src: int) -> Tuple[Dict[int, float], Dict[int, int]]:
    """
    Dijkstra (paso a paso, bien explícito)

    Idea:
      - Mantengo dist[u] = mejor distancia conocida de src a u.
      - 'closed' guarda nodos ya fijados (distancia final).
      - 'pq' es una cola de prioridad (min-heap) con tuplas (distancia, nodo).

    Pasos:
      1) Inicializo dist[u] = +inf para TODOS los nodos (acá uso infinito).
      2) Seteo dist[src] = 0.
      3) Pongo (0, src) en la cola 'pq'  ← acá “agrego el primer nodo”.
      4) Mientras haya elementos en 'pq':
           a) saco el (d, u) más chico
           b) si u ya está cerrado, sigo
           c) cierro u (su distancia final es d)
           d) para cada arista u->v con peso w, intento relajar:
                 nd = d + w
                 si nd < dist[v]: actualizo dist[v] y empujo (nd, v) a 'pq'
    """
    INF = float("inf")

    # (1) Inicializo todo en +inf
    dist: Dict[int, float] = {u: INF for u in g.nodes}
    parent: Dict[int, int] = {}

    # (2) Origen en 0
    dist[src] = 0.0

    # (3) Agrego el primer nodo a la cola
    pq: List[Tuple[float, int]] = [(0.0, src)]
    closed = set()

    # (4) Bucle principal
    while pq:
        d, u = heapq.heappop(pq)
        if u in closed:
            continue

        # cierro u: su mejor distancia ya es definitiva
        closed.add(u)

        # relajo aristas salientes
        for v, w in g.neighbors(u):
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))

    return dist, parent


def reconstruir_camino(parent: Dict[int, int], src: int, dst: int):
    """Reconstruye el camino src->dst usando 'parent' (si existe)."""
    if src == dst:
        return [src]
    if dst not in parent:
        return None  # no hay camino
    path = [dst]
    while path[-1] != src:
        path.append(parent[path[-1]])
    path.reverse()
    return path





class SPCache:
    def __init__(self, g: Graph):
        self.g = g
        self.cache: Dict[int, Dict[int, float]] = {}
    def from_src(self, src: int) -> Dict[int, float]:
        if src not in self.cache:
            self.cache[src] = dijkstra_verbose(self.g, src)[0]
        return self.cache[src]
        # azúcar sintáctico útil
    def distance(self, a: int, b: int) -> float:
        return self.from_src(a).get(b, float("inf"))

# ---- Índice auxiliar (si no lo tenés ya) ----
def index_workers_by_node(workers: List[Worker]) -> Dict[int, List[int]]:
    idx = defaultdict(list)
    for k, w in enumerate(workers):
        idx[w.home].append(k)
    return dict(idx)
#A cada worker le asigas un indice y te dice qe workers vive en cada nodo
#[,...,{2,3,54}] significa que e el nodo i esta los workers 2,3 y 54
#home es un NODO donde esta un worker k

# 1) y 2) Cobertura con radio: v -> [workers]  y  worker -> [nodos]
# =========================================================
def build_worker_cover_maps(
    g: Graph, workers: List[Worker]
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """
    Devuelve:
      - trabajadores_por_nodo[v]: lista de índices de trabajadores i que pueden llegar a v (d(vi, v) <= r_i).
      - alcanzables_por_worker[i]: lista de nodos v alcanzables desde home_i respetando r_i.
    Optimiza calculando Dijkstra una sola vez por nodo hogar distinto.
    """
    sp = SPCache(g)
    by_home = index_workers_by_node(workers)

    trabajadores_por_nodo: Dict[int, List[int]] = defaultdict(list)
    alcanzables_por_worker: Dict[int, List[int]] = {}

    for home, w_ids in by_home.items():#REvisar que no este pasando por nodos e los que no hay workers. Tal vez hay que pasar, pero tenrelo en cuenta.
        dist = sp.from_src(home)  # d(home, ·) para este hogar te da el Dijkstra para el nodo home, lista celeste de antes
        for i in w_ids:
            r = workers[i].r
            nodos = [v for v, d in dist.items() if d <= r]
            nodos.sort() #Ni idea pq
            alcanzables_por_worker[i] = nodos #Nodos me dice todos los nodos que puede alcanzar el worker i
            for v in nodos:
                trabajadores_por_nodo[v].append(i)# En cada iteracion le agrego a cada nodo v los trabajadores que pueden llegar a v

    # ordenar para determinismo
    for v in trabajadores_por_nodo:
        trabajadores_por_nodo[v].sort()

    return dict(trabajadores_por_nodo), alcanzables_por_worker

# ============================================
# 3) Reachability del grafo: u -> [nodos]
# ============================================
def build_graph_reachability(g: Graph) -> Dict[int, List[int]]:
    """
    Para cada nodo u, lista los nodos alcanzables (con distancia finita) en el grafo dirigido.
    No hay umbral de distancia acá.
    """
    sp = SPCache(g)
    return {u: sorted(sp.from_src(u).keys()) for u in g.nodes}




def run_blast_selection_dynamic(
    home: int,
    g: Graph,
    trabajadores_por_nodo: Dict[int, List[int]],
    alcanzables_por_worker: Dict[int, List[int]],
    current_pos: Optional[int] = None,
):
    """
    PASO 1: calcular Dijkstra desde la posición actual (arranca en `home`).
    - No valida nada, asume entradas correctas.
    - Devuelve temprano lo que vamos a usar en los siguientes pasos.
    """
    pos = home if current_pos is None else current_pos

    # PASO 1: Dijkstra desde `pos`
    dist, parent = dijkstra_verbose(g, pos)

    # RETURN TEMPRANO (solo paso 1)
    return {
        "pos": pos,            # posición desde la cual corrimos Dijkstra
        "dist": dist,          # distancias mínimas desde `pos` a todos
        "parent": parent,      # árbol de predecesores para reconstruir caminos
        # A PARTIR DE ACÁ iremos agregando más campos en pasos siguientes.
        # Ej: 'candidatos', 'heap', 'eleccion', 'orden', 'costo', etc.

        
    }
if __name__ == "__main__":
    G_PATH = "dataset/dataset_enviar/grafo.csv"
    W_PATH = "dataset/dataset_enviar/instancia1.csv"

    g = read_graph_csv(G_PATH, directed=True, skip_header=False)
    workers = read_workers_csv(W_PATH, skip_header=False)
    TpN, AporW = build_worker_cover_maps(g, workers)

    out = run_blast_selection_dynamic(home=0, g=g,
                                      trabajadores_por_nodo=TpN,
                                      alcanzables_por_worker=AporW)

    print("POS:", out["pos"])
    # ejemplo: mostrar distancia a los primeros 5 nodos con gente
    for v in list(TpN.keys())[:5]:
        print(f"dist({out['pos']} -> {v}) = {out['dist'].get(v)}")


"""
#revisar que conjuntos se esten formando bien
def guardar_mapas_txt(filepath: str,
                      trabajadores_por_nodo: Dict[int, List[int]],
                      alcanzables_por_worker: Dict[int, List[int]]) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("=== trabajadores_por_nodo ===\n")
        for v in sorted(trabajadores_por_nodo):
            f.write(f"{v}: {trabajadores_por_nodo[v]}\n")

        f.write("\n=== alcanzables_por_worker ===\n")
        for i in sorted(alcanzables_por_worker):
            f.write(f"{i}: {alcanzables_por_worker[i]}\n")


if __name__ == "__main__":
    G_PATH = "dataset/dataset_enviar/grafo.csv"
    W_PATH = "dataset/dataset_enviar/instancia1.csv"

    g = read_graph_csv(G_PATH, directed=True, skip_header=False)
    workers = read_workers_csv(W_PATH, skip_header=False)

    trabajadores_por_nodo, alcanzables_por_worker = build_worker_cover_maps(g, workers)

    guardar_mapas_txt("mapas_cobertura.txt",
                      trabajadores_por_nodo,
                      alcanzables_por_worker)

    print('Listo: guardé los mapas en "mapas_cobertura.txt"')
# =========================
# Ejemplo mínimo
# =========================
if __name__ == "__main__":
    G_PATH = "dataset/dataset_enviar/grafo.csv"
    W_PATH = "dataset/dataset_enviar/instancia1.csv"

    g = read_graph_csv(G_PATH, directed=True, skip_header=False)
    workers = read_workers_csv(W_PATH, skip_header=False)

    trab_por_nodo, nodos_por_worker = build_worker_cover_maps(g, workers)
    print(f"Trabajadores que pueden llegar a 0: {trab_por_nodo.get(0, [])}")
    print(f"Nodos que puede pisar el worker 0 (respetando r0): {nodos_por_worker.get(0, [])[:10]} ...")

    # (opcional) reachability completo de grafo:
    # alc = build_graph_reachability(g)
    # print(f"Nodos alcanzables desde 0: {alc.get(0, [])[:10]} ...")

# A PAARTIR DE ACA, ES PARA VER SI LAS LSITAS SE CREARON BIEN(revisandolo medio a ojo)
    # ---------- Vecinos directos (salientes y entrantes) ----------
def successors_of(g: Graph, v: int):
    return sorted([(u, w) for u, w in g.neighbors(v)], key=lambda x: x[0])

def predecessors_of(g: Graph, v: int):
    preds = []
    for u, adj in g.edges.items():
        if v in adj:
            preds.append((u, adj[v]))
    return sorted(preds, key=lambda x: x[0])

# ---------- Reporte de reachability de trabajadores a un objetivo ----------
def print_workers_reaching_target(g: Graph, workers: list[Worker], target: int = 0, show_non_reachers: bool = True):
    print(f"\n=== Chequeo para target = {target} ===")

    # 1) Vecinos directos del grafo
    succ = successors_of(g, target)
    preds = predecessors_of(g, target)
    print("\nVecinos SALIENTES de {0} ( {0} -> u ):".format(target))
    for u, w in succ:
        print(f"  {target} -> {u} (costo {w})")
    print("\nVecinos ENTRANTES a {0} ( u -> {0} ):".format(target))
    for u, w in preds:
        print(f"  {u} -> {target} (costo {w})")

    # 2) Trabajadores que pueden llegar a 'target' (d(home, target) <= r_i)
    sp = SPCache(g)
    reachers = []
    non_reachers = []
    for i, w in enumerate(workers):
        d = sp.distance(w.home, target)
        if d <= w.r:
            reachers.append((i, w.home, w.r, d))
        else:
            non_reachers.append((i, w.home, w.r, d))

    reachers.sort(key=lambda t: (t[3], t[1]))  # ordeno por distancia y luego por nodo hogar
    print(f"\nTrabajadores que PUEDEN llegar a {target} (d <= r_i): {len(reachers)} / {len(workers)}")
    for i, home, r, d in reachers:
        print(f"  worker {i:>3} | home={home:>5} | r_i={r:.3f} | dist({home}->{target})={d:.3f}  -> OK")

    if show_non_reachers and non_reachers:
        print(f"\n(Info) Algunos que NO llegan a {target} (primeros 10):")
        for i, home, r, d in non_reachers[:10]:
            gap = d - r
            print(f"  worker {i:>3} | home={home:>5} | r_i={r:.3f} | dist={d:.3f}  -> EXCEDE por {gap:.3f}")

# =========================
# Ejemplo de uso en __main__
# =========================
if __name__ == "__main__":
    G_PATH = "dataset/dataset_enviar/grafo.csv"
    W_PATH = "dataset/dataset_enviar/instancia1.csv"

    g = read_graph_csv(G_PATH, directed=True, skip_header=False)
    workers = read_workers_csv(W_PATH, skip_header=False)

    # Si usás tu SPCache con dijkstra_verbose, perfecto; también funciona con dijkstra_lazy
    print_workers_reaching_target(g, workers, target=0, show_non_reachers=True)

    """
