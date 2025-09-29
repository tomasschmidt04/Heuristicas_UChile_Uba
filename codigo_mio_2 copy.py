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




    # (opcional) reachability completo de grafo:
    # alc = build_graph_reachability(g)
    # print(f"Nodos alcanzables desde 0: {alc.get(0, [])[:10]} ...")





# =========================
# BLAST dinámico (distancia desde la posición actual)
# =========================
import random
import heapq
from typing import Dict, List, Tuple, Set, Optional

def _to_set_map(trabajadores_por_nodo: Dict[int, List[int]]) -> Dict[int, Set[int]]:
    return {v: set(ws) for v, ws in trabajadores_por_nodo.items()}


def _make_heap(dist_curr: Dict[int, float],
               TpN: Dict[int, Set[int]],
               alpha: float,
               beta: float,
               dist_scale: float,
               count_scale: float,
               excluidos: Set[int]) -> List[Tuple[float, int, float, int]]:
    """
    Heap de (score, v, dist, cant) SOLO con nodos:
      - que aún tengan gente,
      - alcanzables desde la posición actual (dist < inf),
      - no visitados (no en 'excluidos').
    Score normalizado: alpha*(d/dist_scale) - beta*(cnt/count_scale)
    """
    h: List[Tuple[float, int, float, int]] = []
    for v, workers in TpN.items():
        if not workers or v in excluidos:
            continue
        d = dist_curr.get(v, float("inf"))
        if not math.isfinite(d):
            continue
        cnt = len(workers)
        score = alpha * (d / (dist_scale or 1.0)) - beta * (cnt / (count_scale or 1.0))
        heapq.heappush(h, (score, v, d, cnt))
    return h

def _pick_topk_random(h: List[Tuple[float, int, float, int]],
                      k: int,
                      rng: random.Random) -> Optional[Tuple[float, int, float, int]]:
    if not h:
        return None
    k = min(k, len(h))
    candidates = [heapq.heappop(h) for _ in range(k)]
    return rng.choice(candidates)

def _remove_workers_from_TpN(
    TpN: Dict[int, Set[int]],
    alcanzables_por_worker: Dict[int, List[int]],
    picked_node: int,
    nodos_vetados: Set[int],
    home: int
) -> int:
    """
    Levanta TODOS los workers en 'picked_node' y los elimina de todos los nodos alcanzables.
    Si algún nodo queda vacío -> pasa a 'nodos_vetados' (excepto 'home') y se borra de TpN.
    Devuelve cuántos levantó en 'picked_node'.
    """
    removed = list(TpN.get(picked_node, set()))
    if not removed:
        return 0

    # para cada worker levantado, limpiarlo de todos los nodos a los que podía llegar
    for w in removed:
        for u in alcanzables_por_worker.get(w, []):
            ws = TpN.get(u)
            if ws is None:
                continue
            ws.discard(w)
            if not ws:
                if u != home:
                    nodos_vetados.add(u)
                del TpN[u]

    return len(removed)


def estimate_scales(
    home: int,
    g: Graph,
    trabajadores_por_nodo: Dict[int, List[int]]
) -> Tuple[float, float]:
    """
    Estima escalas para normalizar:
      - dist_scale  = mediana de distancias finitas desde 'home' a nodos con gente.
      - count_scale = mediana de |TpN[v]| (número de personas por nodo, >0).
    """
    sp = SPCache(g)
    dist_home = sp.from_src(home)
    dvals = [dist_home[v] for v, ws in trabajadores_por_nodo.items()
             if ws and math.isfinite(dist_home.get(v, float("inf")))]
    cvals = [len(ws) for ws in trabajadores_por_nodo.values() if len(ws) > 0]
    dist_scale = statistics.median(dvals) if dvals else 1.0
    count_scale = statistics.median(cvals) if cvals else 1.0
    return dist_scale, count_scale

def run_blast_selection_dynamic(
    home: int,
    g: Graph,
    trabajadores_por_nodo: Dict[int, List[int]],
    alcanzables_por_worker: Dict[int, List[int]],
    alpha: float,
    beta: float,
    topk: int = 3,
    seed: int = 123,
    return_home: bool = True,
    dist_scale: Optional[float] = None,
    count_scale: Optional[float] = None,
    nodos_vetados: Optional[Set[int]] = None,  # vetados para selección (transitables)
    require_people: bool = True,
    pickup_at_home: bool = True,
) -> Tuple[List[int], int, float, float]:
    rng = random.Random(seed)
    TpN_sets = {v: set(ws) for v, ws in trabajadores_por_nodo.items()}
    nodos_vetados = set(nodos_vetados or set())

    # escalas fijas iniciales (podés usar re-escalado por iteración si te conviene)
    if dist_scale is None or count_scale is None:
        dist_scale, count_scale = estimate_scales(home, g, trabajadores_por_nodo)

    pos_curr = home
    visitados: Set[int] = set([home])
    orden: List[int] = []
    total_levantados = 0
    tour_cost = 0.0

    sp = SPCache(g)

    # recoger en home (costo 0)
    if pickup_at_home and TpN_sets.get(home):
        total_levantados += _remove_workers_from_TpN(
            TpN_sets, alcanzables_por_worker, home, nodos_vetados, home
        )

    while True:
        dist_curr = sp.from_src(pos_curr)  # Dijkstra normal

        # heap: solo nodos con gente, NO visitados y NO vetados
        h: List[Tuple[float, int, float, int]] = []
        for v, ws in TpN_sets.items():
            if v in visitados or v in nodos_vetados:
                continue
            cnt = len(ws)
            if require_people and cnt == 0:
                continue
            d = dist_curr.get(v, float("inf"))
            if not math.isfinite(d):
                continue
            score = alpha * (d / (dist_scale or 1.0)) - beta * (cnt / (count_scale or 1.0))
            heapq.heappush(h, (score, v, d, cnt))

        if not h:
            break

        k = min(topk, len(h))
        candidates = [heapq.heappop(h) for _ in range(k)]
        score, v, d, cnt = rng.choice(candidates)

        # levantar; si “se vació” por efectos cruzados, NO avanzamos ni costeamos
        levantados = _remove_workers_from_TpN(
            TpN_sets, alcanzables_por_worker, v, nodos_vetados, home
        )
        if require_people and levantados == 0:
            continue

        tour_cost += d
        total_levantados += levantados
        orden.append(v)
        pos_curr = v
        visitados.add(v)

    tour_cost_with_return = tour_cost
    if return_home and orden:
        d_back = sp.distance(pos_curr, home)  # real (transitando vetados si hace falta)
        if math.isfinite(d_back):
            tour_cost_with_return += d_back

    return orden, total_levantados, tour_cost, tour_cost_with_return

# =========================
# Ejemplo de uso en __main__
# ======================



def compute_route_cost(sp: SPCache, home: int, route: List[int], return_home: bool = True) -> float:
    """
    Costo = suma de dist(pos->siguiente) usando distancias mínimas.
    Si alguna pata es inalcanzable, devuelve +inf.
    """
    if not route:
        return 0.0 if not return_home else 0.0  # ruta vacía = costo 0

    cost = 0.0
    curr = home
    for v in route:
        d = sp.distance(curr, v)
        if not math.isfinite(d):
            return float("inf")
        cost += d
        curr = v

    if return_home and route:
        d_back = sp.distance(curr, home)
        if not math.isfinite(d_back):
            return float("inf")
        cost += d_back

    return cost


def two_opt_improve(
    home: int,
    g: Graph,
    route: List[int],
    return_home: bool = True,
    max_iters: int = 50,
    first_improvement: bool = True,
) -> Tuple[List[int], float]:
    """
    2-opt sobre la secuencia de nodos (no toca 'home').
    - Reversa segmentos route[i:j+1] cuando mejora el costo.
    - En cada evaluación usa distancias mínimas (SPCache).
    - Si un reorden genera alguna pata inalcanzable, se descarta.
    Devuelve (ruta_mejorada, costo).
    """
    if len(route) < 2:
        # nada que mejorar
        sp = SPCache(g)
        return route, compute_route_cost(sp, home, route, return_home)

    sp = SPCache(g)
    best_route = route[:]
    best_cost = compute_route_cost(sp, home, best_route, return_home)
    if not math.isfinite(best_cost):
        # por si acaso, aunque la ruta original debería ser alcanzable
        return route, best_cost

    n = len(route)
    it = 0
    improved = True

    while improved and it < max_iters:
        improved = False
        it += 1
        for i in range(0, n - 1):
            for j in range(i + 1, n):
                # Candidato: invertir el segmento [i, j]
                cand = best_route[:i] + best_route[i:j + 1][::-1] + best_route[j + 1:]
                cand_cost = compute_route_cost(sp, home, cand, return_home)
                if cand_cost + 1e-9 < best_cost:
                    best_route, best_cost = cand, cand_cost
                    improved = True
                    if first_improvement:
                        break
            if improved and first_improvement:
                break

    return best_route, best_cost


# =========================
# Wrapper: BLAST + 2-opt
# =========================
def run_blast_with_2opt(
    home: int,
    g: Graph,
    trabajadores_por_nodo: Dict[int, List[int]],
    alcanzables_por_worker: Dict[int, List[int]],
    alpha: float,
    beta: float,
    topk: int = 3,
    seed: int = 123,
    return_home: bool = True,
) -> Tuple[List[int], float, List[int], float, int]:
    """
    1) Corre BLAST dinámico para obtener 'orden' y 'costo' (con regreso).
    2) Aplica 2-opt sobre 'orden'.
    Devuelve:
      - orden_inicial, costo_inicial,
      - orden_mejorado, costo_mejorado,
      - total_trabajadores_levantados
    """
    # BLAST dinámico (tu función existente)
    orden, levantados, c_no_ret, c_ret = run_blast_selection_dynamic(
        home=home,
        g=g,
        trabajadores_por_nodo=trabajadores_por_nodo,
        alcanzables_por_worker=alcanzables_por_worker,
        alpha=alpha,
        beta=beta,
        topk=topk,
        seed=seed,
        return_home=return_home,
    )

    # 2-opt sobre la secuencia de nodos elegidos
    orden_2opt, costo_2opt = two_opt_improve(
        home=home,
        g=g,
        route=orden,
        return_home=return_home,
        max_iters=50,
        first_improvement=True,
    )

    return orden, c_ret, orden_2opt, costo_2opt, levantados
def materializar_tour(g: Graph, home: int, orden: List[int], return_home: bool = True):
    """
    Convierte la secuencia de paradas `orden` en un camino explícito usando caminos mínimos.
    Devuelve: full_path (lista de nodos consecutivos), costo_total, legs (detalle de cada tramo).
    Si algún tramo es inalcanzable, lanza ValueError.
    """
    pos = home
    full_path = [home]
    costo = 0.0
    legs = []  # cada item: {"from": u, "to": v, "cost": d, "path": [u,...,v]}

    for v in orden:
        dist, parent = dijkstra_verbose(g, pos)
        d = dist.get(v, float("inf"))
        if not math.isfinite(d):
            raise ValueError(f"No hay camino desde {pos} a {v}.")
        path = reconstruir_camino(parent, pos, v)
        legs.append({"from": pos, "to": v, "cost": d, "path": path})
        costo += d
        full_path.extend(path[1:])  # evito duplicar 'pos'
        pos = v

    if return_home and orden:
        dist, parent = dijkstra_verbose(g, pos)
        d = dist.get(home, float("inf"))
        if not math.isfinite(d):
            raise ValueError(f"No hay camino de regreso desde {pos} a {home}.")
        path = reconstruir_camino(parent, pos, home)
        legs.append({"from": pos, "to": home, "cost": d, "path": path})
        costo += d
        full_path.extend(path[1:])

    return full_path, costo, legs


def guardar_tour_txt(filepath: str, home: int, orden: List[int],
                     full_path: List[int], costo_total: float, legs: List[dict],
                     titulo: str = "TOUR"):
    """
    Escribe un .txt con paradas, camino expandido y costos por tramo.
    """
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"{titulo}\n")
        f.write("=" * len(titulo) + "\n\n")

        # Paradas (arrancando en 0)
        f.write("Paradas (con 0 al inicio):\n")
        f.write(", ".join(map(str, [home] + orden)) + "\n\n")

        # Camino expandido
        f.write("Camino expandido (nodo por nodo):\n")
        f.write(", ".join(map(str, full_path)) + "\n\n")

        # Tramos
        f.write("Tramos:\n")
        for k, leg in enumerate(legs, 1):
            u, v, d, path = leg["from"], leg["to"], leg["cost"], leg["path"]
            f.write(f"  {k:02d}) {u} -> {v} | costo = {d:.3f} | via: {', '.join(map(str, path))}\n")
        f.write("\n")

        # Resumen
        f.write(f"Costo total (incluye regreso si aplica): {costo_total:.3f}\n")
        f.write(f"N° de paradas (sin contar el 0): {len(orden)}\n")
        f.write(f"Largo camino expandido (nodos): {len(full_path)}\n")


if __name__ == "__main__":
    # RUTAS DE ARCHIVOS (ajustá si hace falta)
    G_PATH = "dataset/dataset_enviar/grafo.csv"
    W_PATH = "dataset/dataset_enviar/instancia1.csv"

    # ==== CARGA ====
    g = read_graph_csv(G_PATH, directed=True, skip_header=False)
    workers = read_workers_csv(W_PATH, skip_header=False)

    # ==== MAPAS DE COBERTURA + NODOS VETADOS INICIALES ====
    home = 0
    TpN, AporW, nodos_vetados_ini = build_worker_cover_maps(g, workers, home=home)

    # ==== PARÁMETROS BLAST ====
    alpha, beta = 1.0, 1.5   # razonables con normalización
    topk, seed = 3, 42

    # ==== BLAST DINÁMICO (con vetados) ====
    orden, levantados, c_no_ret, c_ret = run_blast_selection_dynamic(
        home=home,
        g=g,
        trabajadores_por_nodo=TpN,
        alcanzables_por_worker=AporW,
        alpha=alpha,
        beta=beta,
        topk=topk,
        seed=seed,
        return_home=True,
        nodos_vetados=nodos_vetados_ini,   # vetados solo para selección (transitables en Dijkstra)
        require_people=True,
        pickup_at_home=True,
    )

    # ==== COSTOS INICIALES ====
    # (ya vienen de la función; acá solo los mostramos claramente)
    print("=== BLAST: COSTOS ===")
    print(f"Nodos en orden (sin incluir 0): {len(orden)}")
    print(f"Trabajadores levantados: {levantados}")
    print(f"Costo sin regreso: {c_no_ret:.3f}")
    print(f"Costo con regreso: {c_ret:.3f}")

    # ==== 2-OPT SOBRE EL ORDEN ====
    orden_2opt, costo_2opt = two_opt_improve(
        home=home,
        g=g,
        route=orden,            # importante: 2-opt trabaja **sin** el home en la lista
        return_home=True,
        max_iters=50,
        first_improvement=True,
    )

    # Para comparar también sin regreso (opcional)
    sp = SPCache(g)
    costo_2opt_sin = compute_route_cost(sp, home, orden_2opt, return_home=False)

    # ==== PRINTS FINALES ====
    print("\n=== 2-OPT: COSTOS ===")
    print(f"Nodos en orden 2-opt (sin incluir 0): {len(orden_2opt)}")
    print(f"Costo 2-opt sin regreso: {costo_2opt_sin:.3f}")
    print(f"Costo 2-opt con regreso: {costo_2opt:.3f}")
    print(f"\nMejora (con regreso): {c_ret:.3f}  ->  {costo_2opt:.3f}  (Δ = {c_ret - costo_2opt:.3f})")
        # ==== DUMP A TXT: TOUR INICIAL ====
    try:
        full_path_ini, costo_tot_ini, legs_ini = materializar_tour(g, home, orden, return_home=True)
        guardar_tour_txt("tour_inicial.txt", home, orden, full_path_ini, costo_tot_ini, legs_ini,
                         titulo="TOUR INICIAL (BLAST)")
        print('✔ Guardado "tour_inicial.txt"')
    except ValueError as e:
        print(f"[WARN] No se pudo materializar el tour inicial: {e}")

    # ==== DUMP A TXT: TOUR 2-OPT ====
    try:
        full_path_2opt, costo_tot_2opt, legs_2opt = materializar_tour(g, home, orden_2opt, return_home=True)
        guardar_tour_txt("tour_2opt.txt", home, orden_2opt, full_path_2opt, costo_tot_2opt, legs_2opt,
                         titulo="TOUR 2-OPT (Mejorado)")
        print('✔ Guardado "tour_2opt.txt"')
    except ValueError as e:
        print(f"[WARN] No se pudo materializar el tour 2-opt: {e}")


    # (Opcional) ver la ruta “con 0 al principio” para inspección visual:
    # print("Ruta inicial (con 0):", [home] + orden[:20], "...")
    # print("Ruta 2-opt   (con 0):", [home] + orden_2opt[:20], "...")
