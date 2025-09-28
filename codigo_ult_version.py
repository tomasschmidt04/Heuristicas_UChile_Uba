# -*- coding: utf-8 -*-
import math, random
from typing import Dict, List, Tuple, Set
from importlib.machinery import SourceFileLoader
from pathlib import Path

# --- helpers ---
mod_path = "codigo_mio_2.py"  # <- uso ruta local simple
mod = SourceFileLoader("codigo_mio_2", mod_path).load_module()

Graph = mod.Graph
Worker = mod.Worker
read_graph_csv = mod.read_graph_csv
read_workers_csv = mod.read_workers_csv
build_worker_cover_maps = mod.build_worker_cover_maps
reconstruir_camino = mod.reconstruir_camino
dijkstra_std = mod.dijkstra_verbose

def remove_workers_from_TpN(TpN: Dict[int, Set[int]], AporW: Dict[int, List[int]], picked_node: int) -> int:
    picked = list(TpN.get(picked_node, set()))
    if not picked:
        return 0
    for w in picked:
        for v in AporW.get(w, []):
            if v in TpN:
                TpN[v].discard(w)
                if not TpN[v]:
                    del TpN[v]
    return len(picked)

def greedy_tsp_transit_ok(g: Graph, TpN_list: Dict[int, List[int]], AporW: Dict[int, List[int]],
                          home: int = 0, topk: int = 3, seed: int = 123):
    rng = random.Random(seed)
    TpN: Dict[int, Set[int]] = {v: set(ws) for v, ws in TpN_list.items()}

    visited_stops: Set[int] = {home}
    curr = home
    stops: List[int] = []
    full_path: List[int] = [home]
    logs: List[str] = []

    if home in TpN and TpN[home]:
        got = remove_workers_from_TpN(TpN, AporW, home)
        logs.append(f"[ini] Levanté {got} trabajadores en el nodo 0. Restan={sum(len(s) for s in TpN.values())}")

    it = 0
    while True:
        remaining = sum(len(s) for s in TpN.values())
        if remaining == 0:
            logs.append("[fin] Todos los trabajadores fueron cubiertos.")
            break

        it += 1
        dist, parent = dijkstra_std(g, curr)

        candidates = []
        for v, ws in TpN.items():
            if v in visited_stops:
                continue
            d = dist.get(v, float("inf"))
            if math.isfinite(d):
                candidates.append((d, v))

        if not candidates:
            logs.append(f"[it {it}] No hay candidatos alcanzables con Dijkstra estándar. Me detengo.")
            break

        candidates.sort(key=lambda x: x[0])
        top = candidates[:min(topk, len(candidates))]
        d_chosen, v_chosen = rng.choice(top)

        path = reconstruir_camino(parent, curr, v_chosen)
        if path is None:
            logs.append(f"[it {it}] ERROR: no pude reconstruir camino desde {curr} a {v_chosen}.")
            break

        full_path.extend(path[1:])
        stops.append(v_chosen)
        visited_stops.add(v_chosen)
        curr = v_chosen

        got = remove_workers_from_TpN(TpN, AporW, v_chosen)
        logs.append(f"[it {it}] Elegí {v_chosen} (dist={d_chosen:.3f}). Levanté {got}. Restan={sum(len(s) for s in TpN.values())}")

    if full_path and full_path[-1] != home:
        dist_back, parent_back = dijkstra_std(g, curr)
        if math.isfinite(dist_back.get(home, float("inf"))):
            path_back = reconstruir_camino(parent_back, curr, home)
            full_path.extend(path_back[1:])
            logs.append("[return] Volví a 0 con Dijkstra estándar.")
        else:
            logs.append("[return] No encontré camino de regreso a 0.")

    resultados_line = " ".join(map(str, [home] + stops + [home]))
    completo_line = " ".join(map(str, full_path))
    return resultados_line, completo_line, logs, full_path  # <- devuelvo también la lista

# --- costo simple del camino explícito ---
def path_cost_from_edges(g: Graph, node_list: List[int]) -> float:
    total = 0.0
    for a, b in zip(node_list[:-1], node_list[1:]):
        w = None
        for v, c in g.neighbors(a):
            if v == b:
                w = c
                break
        if w is None:
            raise ValueError(f"No existe arco {a}->{b} en el grafo")
        total += w
    return total

# ---- cache de distancias de camino mínimo entre paradas ----
class DistCache:
    def __init__(self, g: Graph):
        self.g = g
        self.cache: Dict[int, Dict[int, float]] = {}
    def get(self, u: int, v: int) -> float:
        if u not in self.cache:
            dist, _ = dijkstra_std(self.g, u)
            self.cache[u] = dist
        d = self.cache[u].get(v, math.inf)
        return d

# ---- 2-opt sobre la lista de paradas (tour cerrado con 0 al inicio/fin) ----
def two_opt_on_stops(g: Graph, tour: List[int]) -> List[int]:
    # tour: [0, v1, v2, ..., vk, 0]
    if len(tour) <= 4:
        return tour[:]  # nada para mejorar
    dc = DistCache(g)
    improved = True
    best = tour[:]
    while improved:
        improved = False
        # no tocamos el primer y último (0)
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                a, b = best[i - 1], best[i]
                c, d = best[j], best[j + 1]
                old = dc.get(a, b) + dc.get(c, d)
                new = dc.get(a, c) + dc.get(b, d)
                if new + 1e-12 < old:
                    # invertir el segmento [i, j]
                    best[i:j + 1] = reversed(best[i:j + 1])
                    improved = True
    return best

# ---- reconstruir camino completo a partir de paradas ----
def build_full_path_from_stops(g: Graph, stops_closed: List[int]) -> List[int]:
    full = [stops_closed[0]]
    for a, b in zip(stops_closed[:-1], stops_closed[1:]):
        _, parent = dijkstra_std(g, a)
        tramo = reconstruir_camino(parent, a, b)
        if tramo is None:
            raise ValueError(f"Sin camino entre {a} y {b} al reconstruir post 2-opt")
        full.extend(tramo[1:])
    return full

if __name__ == "__main__":
    # Datos (ajustá rutas si hace falta)
    G_PATH = "dataset/dataset_enviar/grafo.csv"
    W_PATH = "dataset/dataset_enviar/instancia1.csv"
    g = read_graph_csv(G_PATH, directed=True, skip_header=False)
    workers = read_workers_csv(W_PATH, skip_header=False)
    TpN, AporW, _ = build_worker_cover_maps(g, workers, home=0)

    # ---- MULTI-ITERACIONES + 2-OPT (simple) ----
    ITERS = 4          # <--- cambiá acá (p. ej. 10000)
    TOPK  = 3
    BASE_SEED = 123

    best_cost = float("inf")
    best_seed = None
    best_resultados = ""
    best_completo = ""
    best_logs = []
    best_full_list: List[int] = []

    for i in range(ITERS):
        seed = BASE_SEED + i
        resultados_line, completo_line, logs, full_list = greedy_tsp_transit_ok(
            g, TpN, AporW, home=0, topk=TOPK, seed=seed
        )
        # 2-opt sobre paradas
        stops_closed = list(map(int, resultados_line.split()))
        stops_opt = two_opt_on_stops(g, stops_closed)
        # reconstruyo camino completo con el orden optimizado
        full_opt = build_full_path_from_stops(g, stops_opt)
        # costo del camino explícito
        cost = path_cost_from_edges(g, full_opt)

        if cost < best_cost:
            best_cost = cost
            best_seed = seed
            best_resultados = " ".join(map(str, stops_opt))
            best_completo = " ".join(map(str, full_opt))
            best_logs = logs + [f"[2-opt] Mejoró tour con seed={seed}. Costo={cost:.6f}"]
            best_full_list = full_opt

    # ---- imprimir y guardar mejor ----
    outdir = Path("./out")
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "resultados_best.txt").write_text(best_resultados, encoding="utf-8")
    (outdir / "camino_completo_best.txt").write_text(best_completo, encoding="utf-8")
    (outdir / "log_iteraciones_best.txt").write_text("\n".join(best_logs), encoding="utf-8")

    print("\n==== MEJOR CORRIDA (con 2-opt por corrida) ====")
    print(f"seed          = {best_seed}")
    print(f"total_cost    = {best_cost:.6f}")
    print(f"n_stops       = {len(best_resultados.split()) - 2}")
    print(f"n_nodes_full  = {len(best_full_list)}")
    print("Archivos:")
    print(" -", outdir / "resultados_best.txt")
    print(" -", outdir / "camino_completo_best.txt")
    print(" -", outdir / "log_iteraciones_best.txt")
