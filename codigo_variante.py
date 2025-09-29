# -*- coding: utf-8 -*-
"""
Ejercicio 2 (simple, sin movimiento):
- Ya tenemos el tour base del Ej.1 (paradas "0 ... 0").
- Elegimos un subconjunto S de workers a omitir.
- Prohibimos TODOS los nodos alcanzables por esos workers (ni parada ni tránsito).
- Quitamos esos nodos de las paradas base y UNIMOS LOS HUECOS con caminos mínimos
  en el grafo actualizado (Dijkstra evitando prohibidos).
- Costo = suma de arcos del camino materializado.
- Workers servidos = {j : AporW[j] ∩ paradas_finales ≠ ∅}. No hay costo por movimiento.
- P* = (C_base - C_nuevo) / (#no_served).
"""

import math, random, argparse, heapq
from typing import Dict, List, Set, Tuple
from pathlib import Path
from importlib.machinery import SourceFileLoader

# ===== Helpers del Ej.1 =====
def load_helpers(codigo_path: str):
    mod = SourceFileLoader("codigo_mio_2", codigo_path).load_module()
    return (
        mod,                       # módulo completo (por si hace falta)
        mod.Graph,
        mod.read_graph_csv,
        mod.read_workers_csv,
        mod.build_worker_cover_maps,
        mod.reconstruir_camino,
        mod.dijkstra_verbose,      # Dijkstra estándar
    )

# ===== Dijkstra que evita nodos prohibidos (sin mutar el grafo) =====
def dijkstra_avoid(g, src: int, banned: Set[int]):
    INF = float("inf")
    dist = {u: INF for u in g.nodes}
    parent: Dict[int,int] = {}
    dist[src] = 0.0
    pq = [(0.0, src)]
    closed = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in closed: 
            continue
        closed.add(u)
        for v, w in g.neighbors(u):
            if v in banned and v != src:
                continue
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, parent

# ===== Suma de arcos del camino explícito =====
def path_cost_from_edges(g, node_list: List[int]) -> float:
    total = 0.0
    for a, b in zip(node_list[:-1], node_list[1:]):
        w = None
        for v, c in g.neighbors(a):
            if v == b:
                w = c; break
        if w is None:
            raise ValueError(f"No existe arco {a}->{b} en el grafo")
        total += w
    return total

# ===== Parchear el recorrido base "uniendo huecos" =====
def patch_route_by_banning_nodes(
    g,
    base_stops_closed: List[int],   # ej: [0, v1, v2, ..., 0]
    banned: Set[int],
    reconstruir_camino,
    home: int = 0
) -> Tuple[List[int], List[int]]:
    """
    - Quita de las paradas base los nodos prohibidos.
    - Entre par consecutivo de paradas 'permitidas', une con SP evitando prohibidos.
    - Devuelve (stops_final_cerradas, full_path_materializado). Si es inviable, levanta excepción.
    """
    if home in banned:
        raise RuntimeError("Home quedó prohibido por S: inviable.")

    # Paradas que quedan (quitamos prohibidas)
    kept_stops = [s for s in base_stops_closed if s not in banned]

    # Asegurar que arranque y termine en home
    if kept_stops[0] != home:
        kept_stops = [home] + kept_stops
    if kept_stops[-1] != home:
        kept_stops = kept_stops + [home]

    # Si no quedó ninguna parada intermedia, el camino es [0] (costo 0)
    if len(kept_stops) == 2 and kept_stops[0] == kept_stops[1] == home:
        return kept_stops, [home]

    # Materializar camino entre pares consecutivos evitando prohibidos
    full = [kept_stops[0]]
    for a, b in zip(kept_stops[:-1], kept_stops[1:]):
        dist, parent = dijkstra_avoid(g, a, banned)
        if not math.isfinite(dist.get(b, math.inf)):
            raise RuntimeError(f"Inviable unir {a}->{b} evitando prohibidos.")
        tramo = reconstruir_camino(parent, a, b)
        if tramo is None:
            raise RuntimeError(f"Falla reconstrucción {a}->{b}.")
        full.extend(tramo[1:])
    return kept_stops, full

# ===== Conteo de servidos con AporW =====
def count_served_by_stops(AporW: Dict[int, List[int]], stops_closed: List[int], W_total: int) -> int:
    Sset = set(stops_closed)
    served = 0
    for j in range(W_total):
        L = AporW.get(j, [])
        # servido si tiene al menos una parada alcanzable (en el grafo original) que siga en stops
        if any(v in Sset for v in L):
            served += 1
    return served

# ===== Evaluar un subconjunto S =====
def evaluate_subset(
    g, base_stops, C_base,
    AporW: Dict[int, List[int]],
    S_omit: Set[int],
    reconstruir_camino,
    home: int = 0
):
    # Nodos prohibidos: unión de alcanzables de workers en S
    banned: Set[int] = set()
    for w in S_omit:
        banned.update(AporW.get(w, []))

    # Parchear ruta
    stops_S, full_S = patch_route_by_banning_nodes(
        g, base_stops, banned, reconstruir_camino=reconstruir_camino, home=home
    )
    C_new = path_cost_from_edges(g, full_S)

    # No servidos: aquellos cuyo AporW[j] no intersecta con stops_S
    W_total = max(AporW.keys(), default=-1) + 1
    n_served = count_served_by_stops(AporW, stops_S, W_total)
    n_not_served = W_total - n_served
    if n_not_served <= 0:
        return -math.inf, (stops_S, full_S, C_new, banned, n_not_served)

    P_star = (C_base - C_new) / n_not_served
    return P_star, (stops_S, full_S, C_new, banned, n_not_served)

# ===== MAIN =====
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grafo", type=str, default="./dataset/dataset_enviar/grafo.csv")
    ap.add_argument("--instancia", type=str, default="./dataset/dataset_enviar/instancia1.csv")
    ap.add_argument("--codigo", type=str, default="./codigo_mio_2.py")
    ap.add_argument("--base", type=str, default="./out/resultados_best.txt", help="tour base (0 ... 0)")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--kmax", type=int, default=5, help="tamaño máximo del S aleatorio")
    ap.add_argument("--home", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="./out_ej2")
    args = ap.parse_args()

    # Helpers
    mod, Graph, read_graph_csv, read_workers_csv, build_worker_cover_maps, reconstruir_camino, dijkstra_std = load_helpers(args.codigo)

    # Datos + mapeos
    g = read_graph_csv(args.grafo, directed=True, skip_header=False)
    workers = read_workers_csv(args.instancia, skip_header=False)
    # Usamos AporW para (i) construir 'banned' y (ii) contar servidos
    TpN, AporW, _ = build_worker_cover_maps(g, workers, home=args.home)

    # Costo base desde tour del Ej.1
    base_stops = list(map(int, Path(args.base).read_text(encoding="utf-8").split()))
    # materializar full base en grafo original
    full_base = [base_stops[0]]
    for a, b in zip(base_stops[:-1], base_stops[1:]):
        _, parent = dijkstra_std(g, a)
        tramo = reconstruir_camino(parent, a, b)
        if tramo is None:
            raise RuntimeError(f"No hay camino {a}->{b} en tour base")
        full_base.extend(tramo[1:])
    C_base = path_cost_from_edges(g, full_base)
    print(f"[Base] Costo ruta Ej1 = {C_base:.3f} | paradas = {len(base_stops)-1}")

    # Búsqueda aleatoria de S
    rng = random.Random(args.seed)
    W_total = max(AporW.keys(), default=-1) + 1
    all_ids = list(range(W_total))

    best_P = -math.inf
    best = None  # (S, stops_S, full_S, C_new, banned, n_not_served)

    for t in range(args.iters):
        k = rng.randint(1, min(args.kmax, W_total))
        S = set(rng.sample(all_ids, k))
        try:
            P_star, (stops_S, full_S, C_new, banned, n_not_served) = evaluate_subset(
                g, base_stops, C_base, AporW, S, reconstruir_camino, home=args.home
            )
        except RuntimeError:
            continue  # subconjunto inviable (no se puede patchar el tour)

        if P_star > best_P:
            best_P = P_star
            best = (S, stops_S, full_S, C_new, banned, n_not_served)

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    if best is None or best_P <= 0:
        print("\nNo se encontró un S que convenga omitir para P>0.")
    else:
        S, stops_S, full_S, C_new, banned, n_not_served = best
        print("\n====== EJERCICIO 2 ======")
        print(f"P* (máximo) = {best_P:.6f}")
        print(f"Tamaño S (omitidos) = {len(S)} | no_served = {n_not_served}")
        print(f"Costo nuevo = {C_new:.3f} (base {C_base:.3f})")
        print(f"Paradas finales = {len(stops_S)-1}")

        (outdir / "P_star.txt").write_text(f"{best_P:.6f}\n", encoding="utf-8")
        (outdir / "workers_omitidos.txt").write_text(" ".join(map(str, sorted(S))), encoding="utf-8")
        (outdir / "nodos_prohibidos.txt").write_text(" ".join(map(str, sorted(banned))), encoding="utf-8")
        (outdir / "stops_S.txt").write_text(" ".join(map(str, stops_S)), encoding="utf-8")
        (outdir / "full_S.txt").write_text(" ".join(map(str, full_S)), encoding="utf-8")

        print("\nArchivos en:", outdir)

if __name__ == "__main__":
    main()
