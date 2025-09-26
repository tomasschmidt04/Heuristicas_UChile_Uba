
# -*- coding: utf-8 -*-

"""
TP1 - Heurística de ruteo con caminatas (versión SINGLE FILE)

Cómo ejecutar:
    python codigo.py dataset/__MACOSX/dataset_enviar/._grafo.csv dataset/__MACOSX/dataset_enviar/._instancia1.csv --seed 123 --topk 8 --alpha 0.35 --iters 20 --plot 0

- grafo.csv: arcos dirigidos con filas: i,j,cij
- instanciaX.csv: trabajadores con filas: v_i, r_i
- --iters: cantidad de corridas (aleatoriedad). Se guarda la mejor factible.
- --topk / --alpha: controlan la selección tipo BLAST con RCL/GRASP.
- --plot: 1 para graficar el mejor tour (si tenés networkx y matplotlib).

Salida:
- resultados.txt con la ruta en formato: "0 v1 v2 ... vk 0"
"""

import argparse
import csv
import math
import random
from typing import Dict, List, Tuple, Set

# =========================
# Lectura de datos
# =========================
def read_graph(path: str) -> Dict[int, List[Tuple[int, float]]]:
    """Lee grafo dirigido desde CSV con filas: i,j,cij -> adj[nodo] = [(vecino, costo), ...]."""
    adj: Dict[int, List[Tuple[int, float]]] = {}
    nodes = set()
    with open(path, newline="", encoding="utf-8") as fh:
        rd = csv.reader(fh)
        for row in rd:
            if not row or str(row[0]).strip().startswith("#"):
                continue
            i, j, c = int(row[0]), int(row[1]), float(row[2])
            nodes.add(i); nodes.add(j)
            adj.setdefault(i, []).append((j, c))
            adj.setdefault(j, adj.get(j, []))  # asegurar clave aunque no tenga salientes
    for v in nodes:
        adj.setdefault(v, [])
    return adj

def read_instance(path: str) -> List[Tuple[int, float]]:
    """Lee trabajadores: filas 'v_i, r_i' -> lista de (hogar, radio)."""
    workers: List[Tuple[int, float]] = []
    with open(path, newline="", encoding="utf-8") as fh:
        rd = csv.reader(fh)
        for row in rd:
            if not row or str(row[0]).strip().startswith("#"):
                continue
            v, r = int(row[0]), float(row[1])
            workers.append((v, r))
    return workers

# =========================
# Caminos mínimos (Dijkstra)
# =========================
import heapq
def dijkstra(adj: Dict[int, List[Tuple[int, float]]], src: int) -> Dict[int, float]:
    dist = {v: math.inf for v in adj.keys()}
    dist[src] = 0.0
    pq = [(0.0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in adj[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist

def all_pairs_from_sources(adj: Dict[int, List[Tuple[int, float]]], sources) -> Dict[int, Dict[int, float]]:
    """Distancias desde cada src en 'sources' a todos los nodos (Dijkstra)."""
    return {s: dijkstra(adj, s) for s in sources}

# =========================
# Cobertura por nodo
# =========================
def compute_node_coverage(adj, workers: List[Tuple[int, float]], all_pairs) -> Dict[int, Set[int]]:
    """
    cover[v] = {i : trabajador i queda atendido si visitamos el nodo v}
    i está atendido si d(v_i, v) <= r_i
    """
    cover: Dict[int, Set[int]] = {v: set() for v in adj.keys()}
    for i, (home, radius) in enumerate(workers):
        dist_from_home = all_pairs[home]
        for v, d in dist_from_home.items():
            if d <= radius:
                cover[v].add(i)
    return cover

# =========================
# BLAST (sin lookahead)
# =========================
def greedy_score(current: int, cand: int, uncovered: Set[int], cover, dist) -> float:
    """
    Puntaje = nuevos atendidos en cand - penalización leve por distancia actual->cand.
    (Sirve para desempatar hacia caminos más cortos.)
    """
    new_served = len(cover[cand] & uncovered)
    # penal mínimo por distancia para desempatar
    return new_served - 1e-6 * dist[current][cand]

def construct_route(adj, workers, cover, seed=0, topk=8, alpha=0.35):
    """
    Construcción del tour:
      - arranca en depósito (0 si existe, si no, el menor nodo)
      - en cada paso, calcula score para todos los candidatos y elige dentro del top-k
        con umbral RCL controlado por alpha.
      - termina cuando no quedan trabajadores sin cubrir; cierra el ciclo con 0.
    """
    random.seed(seed)
    nodes = list(adj.keys())
    depot = 0 if 0 in adj else min(nodes)
    uncovered = set(range(len(workers)))
    route = [depot]

    # dist[u][v] para todos los u que nos interesan (acá: todos los nodos)
    dist = all_pairs_from_sources(adj, nodes)

    while uncovered:
        scored = []
        cur = route[-1]
        for v in nodes:
            if v == cur:
                continue
            s = greedy_score(cur, v, uncovered, cover, dist)
            if math.isfinite(s):
                # guardo (score, crit_empate, nodo)
                scored.append((s, -dist[cur][v], v))

        if not scored:
            # No hay candidatos (grafo desconexo o radios imposibles)
            break

        scored.sort(reverse=True)
        rcl = scored[:max(1, min(topk, len(scored)))]

        best_s = rcl[0][0]
        worst_s = rcl[-1][0]
        # umbral tipo GRASP: más chico alpha => más voraz
        threshold = best_s - alpha * (best_s - worst_s + 1e-12)
        candidates = [t for t in rcl if t[0] >= threshold]

        chosen = random.choice(candidates)[2]
        route.append(chosen)
        # actualizar cobertura
        uncovered -= cover[chosen]

    # cerrar ciclo
    if route[-1] != depot:
        route.append(depot)
    else:
        route.append(depot)

    return route, dist

# =========================
# 2-Opt (mejora)
# =========================
def path_cost(order: List[int], dist) -> float:
    total = 0.0
    for a, b in zip(order, order[1:]):
        total += dist[a][b]
    return total

def two_opt(order: List[int], dist, max_iters: int = 2000) -> List[int]:
    """
    2-Opt clásico sobre el orden de visita (mantiene extremos y conjunto de nodos).
    """
    best = order[:]
    best_cost = path_cost(best, dist)
    n = len(order)
    improved = True
    it = 0
    while improved and it < max_iters:
        improved = False
        it += 1
        for i in range(1, n - 2):      # no cortar en 0 ni en el último (0 de retorno)
            for k in range(i + 1, n - 1):
                new_order = best[:i] + best[i:k+1][::-1] + best[k+1:]
                new_cost = path_cost(new_order, dist)
                if new_cost + 1e-9 < best_cost:
                    best = new_order
                    best_cost = new_cost
                    improved = True
                    break
            if improved:
                break
    return best

# =========================
# Utilidades
# =========================
def is_feasible(route, cover, workers_count) -> bool:
    covered = set()
    for v in set(route):
        covered |= cover[v]
    return len(covered) == workers_count

def save_result(route: List[int], out_path="resultados.txt"):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(" ".join(map(str, route)) + "\n")

# =========================
# Plot opcional
# =========================
def plot_route(adj, route):
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
    except Exception:
        return
    G = nx.DiGraph()
    for u, nbrs in adj.items():
        for v, w in nbrs:
            G.add_edge(u, v, weight=w)
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8,6))
    nx.draw(G, pos, node_size=50, arrows=False, alpha=0.3)
    path_edges = list(zip(route, route[1:]))
    nx.draw_networkx_nodes(G, pos, nodelist=list(set(route)), node_color="tab:blue", node_size=80)
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=2.0, edge_color="tab:red", arrows=False)
    nx.draw_networkx_labels(G, pos, labels={v: v for v in set(route)}, font_size=8)
    plt.title("Mejor ruta encontrada")
    plt.tight_layout()
    plt.show()

# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grafo_csv")
    ap.add_argument("instancia_csv")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.35)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--plot", type=int, default=0)
    ap.add_argument("--out", type=str, default="resultados.txt")
    args = ap.parse_args()

    random.seed(args.seed)

    # Leer datos
    adj = read_graph(args.grafo_csv)
    workers = read_instance(args.instancia_csv)
    homes = [v for (v, _) in workers]

    # Distancias desde los hogares para cobertura
    all_pairs = all_pairs_from_sources(adj, set(homes))
    cover = compute_node_coverage(adj, workers, all_pairs)

    best_route = None
    best_cost = float("inf")

    # Corridas independientes (aleatoriedad en RCL)
    for it in range(args.iters):
        route, dist = construct_route(
            adj, workers, cover,
            seed=args.seed + it,
            topk=args.topk,
            alpha=args.alpha
        )
        # Mejora 2-Opt manteniendo extremos 0...0
        inner = route[1:-1]
        improved = two_opt([route[0]] + inner + [route[-1]], dist)
        improved_inner = improved[1:-1]
        route = [route[0]] + improved_inner + [route[-1]]

        cost = path_cost(route, dist)
        feas = is_feasible(route, cover, len(workers))

        if feas and cost < best_cost:
            best_cost = cost
            best_route = route

    # Fallback si no encontró factible (poco probable salvo radios imposibles)
    if best_route is None:
        best_route = [0, 0]

    save_result(best_route, args.out)
    print(f"Best cost: {best_cost:.4f}")
    print(f"Route: {' '.join(map(str, best_route))}")
    print(f"Saved to: {args.out}")

    if args.plot:
        plot_route(adj, best_route)

if __name__ == "__main__":
    main()