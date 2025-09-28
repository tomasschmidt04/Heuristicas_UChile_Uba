# -*- coding: utf-8 -*-
import csv
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Iterable, Tuple,Optional,Set
import heapq
import random, math, statistics

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



# 1) y 2) Cobertura con radio: v -> [workers]  y  worker -> [nodos]
# =========================================================
def build_worker_cover_maps(
    g: Graph,
    workers: List[Worker],
    home: Optional[int] = None
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]], Set[int]]:
    """
    Devuelve:
      - trabajadores_por_nodo[v]: lista de índices i que pueden llegar a v (d(vi,v) ≤ r_i)
      - alcanzables_por_worker[i]: nodos v alcanzables por i
      - nodos_vetados_ini: nodos que ningún trabajador puede alcanzar (no aparecen como key)
        (home nunca se veta aquí)
    """
    sp = SPCache(g)
    by_home = index_workers_by_node(workers)

    trabajadores_por_nodo: Dict[int, List[int]] = defaultdict(list)
    alcanzables_por_worker: Dict[int, List[int]] = {}

    for h, w_ids in by_home.items():
        dist = sp.from_src(h)
        for i in w_ids:
            r = workers[i].r
            nodos = [v for v, d in dist.items() if d <= r]
            nodos.sort()
            alcanzables_por_worker[i] = nodos
            for v in nodos:
                trabajadores_por_nodo[v].append(i)

    for v in trabajadores_por_nodo:
        trabajadores_por_nodo[v].sort()

    all_nodes = set(g.nodes)
    covered_nodes = set(trabajadores_por_nodo.keys())
    nodos_vetados_ini: Set[int] = all_nodes - covered_nodes
    if home is not None and home in nodos_vetados_ini:
        nodos_vetados_ini.remove(home)  # home nunca vetado de arranque

    return dict(trabajadores_por_nodo), alcanzables_por_worker, nodos_vetados_ini

