# Heuristicas 
# 2-opt y Nearest Neighbor aleatorizado

import csv
import heapq
import random
import sys
import time 
import math

# ---------------------------
# Leer grafo
# ---------------------------
def load_graph(grafo_file):
    edges = []
    with open(grafo_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            u, v, c = int(row[0]), int(row[1]), float(row[2])
            edges.append((u, v, c))

    if not edges:
        raise ValueError("grafo.csv vacío")

    N = 1 + max(max(u, v) for u, v, _ in edges)  # número de nodos
    # inicializamos matriz NxN con INF
    graph = [[math.inf] * N for _ in range(N)]
    for i in range(N):
        graph[i][i] = 0.0  # costo a sí mismo = 0

    # cargar arcos
    for u, v, c in edges:
        graph[u][v] = c  # si hay múltiple arco, se queda con el último leído

    return N, graph

def remove_nodes(graph, forbidden_nodes):
    N = len(graph)
    new_graph = [row[:] for row in graph]  # copia profunda

    for node in forbidden_nodes:
        for j in range(N):
            new_graph[node][j] = math.inf   # eliminar aristas salientes
            new_graph[j][node] = math.inf   # eliminar aristas entrantes
        new_graph[node][node] = math.inf    # incluso el lazo propio

    return new_graph
# ---------------------------
# Dijkstra: distancias mínimas desde un nodo
# ---------------------------
def dijkstra(graph, start):
    """Dijkstra optimizado usando matriz de adyacencia"""
    n = len(graph)
    dist = [math.inf] * n
    dist[start] = 0
    visited = [False] * n
    
    for _ in range(n):
        # Encontrar nodo no visitado con distancia mínima
        u = -1
        for v in range(n):
            if not visited[v] and (u == -1 or dist[v] < dist[u]):
                u = v
        
        if dist[u] == math.inf:
            break
            
        visited[u] = True
        
        # Actualizar distancias de vecinos
        for v in range(n):
            if not visited[v] and graph[u][v] != math.inf:
                new_dist = dist[u] + graph[u][v]
                if new_dist < dist[v]:
                    dist[v] = new_dist
    
    return {i: dist[i] for i in range(n)}

def dijkstra_with_path(adj_matrix, start, end):
    """Dijkstra que retorna el camino completo - optimizado"""
    if start == end:
        return [start]
    
    n = len(adj_matrix)
    dist = [math.inf] * n
    parent = [-1] * n
    visited = [False] * n
    
    dist[start] = 0
    
    for _ in range(n):
        u = -1
        for v in range(n):
            if not visited[v] and (u == -1 or dist[v] < dist[u]):
                u = v
        
        if u == end or dist[u] == math.inf:
            break
            
        visited[u] = True
        
        for v in range(n):
            if not visited[v] and adj_matrix[u][v] != math.inf:
                new_dist = dist[u] + adj_matrix[u][v]
                if new_dist < dist[v]:
                    dist[v] = new_dist
                    parent[v] = u
    
    # Reconstruir camino
    if parent[end] == -1 and start != end:
        return [start, end]  # No hay camino
    
    path = []
    current = end
    while current != -1:
        path.append(current)
        current = parent[current]
    
    path.reverse()
    return path

# ---------------------------
# Leer instancia (trabajadores)
# ---------------------------
def load_instance(instancia_file):
    workers = []
    with open(instancia_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            vi, ri = int(row[0]), float(row[1])
            workers.append((vi, ri))
    return workers

# ---------------------------
# Construcción inicial
# ---------------------------
def build_initial_solution(graph, workers, depot=0, strategy='greedy'):
    """
    Estrategias mejoradas:
    - 'random': aleatorio (original)
    - 'greedy': más cercano al depósito
    - 'center': más central (suma mínima de distancias)
    - 'mixed': combina estrategias
    """
    N = len(graph)
    all_dist = {}
    all_dist[depot] = dijkstra(graph, depot)
    Ci = []
    meet_points = []
    
    for i, (vi, ri) in enumerate(workers):
        print(f"Procesando trabajador {i+1}/{len(workers)} en nodo {vi} con radio {ri}")
        all_dist[vi] = dijkstra(graph, vi)
        reachable = [v for v in range(N) if all_dist[vi][v] <= ri]
        Ci.append(reachable)

        if depot in reachable:
            chosen = depot
        elif vi not in [mp for mp in meet_points]:
            if strategy == 'random':
                chosen = random.choice(reachable)
            elif strategy == 'greedy':
                chosen = min(reachable, key=lambda v: all_dist[depot][v])
            elif strategy == 'center':
                # Punto más central: minimiza suma de distancias a todos los nodos
                for v in reachable:
                    if v not in all_dist:
                        all_dist[v] = dijkstra(graph, v)
                chosen = min(reachable, key=lambda v: sum(all_dist[v].values()))
            elif strategy == 'mixed':
                # Alterna entre estrategias
                if i % 3 == 0:
                    chosen = min(reachable, key=lambda v: all_dist[depot][v])
                elif i % 3 == 1:
                    chosen = random.choice(reachable)
                else:
                    for v in reachable:
                        if v not in all_dist:
                            all_dist[v] = dijkstra(graph, v)
                    chosen = min(reachable, key=lambda v: sum(all_dist[v].values()))
            
            if chosen not in meet_points:
                meet_points.append(chosen)
                if chosen not in all_dist:
                    all_dist[chosen] = dijkstra(graph, chosen)
    
    # Verificar cobertura
    uncovered = 0
    for (vi, ri) in workers:
        reachable = [v for v, d in all_dist[vi].items() if d <= ri]
        if not any(mp in reachable for mp in meet_points):
            uncovered += 1
    
    if uncovered > 0:
        print(f"⚠️ {uncovered} trabajadores no quedaron cubiertos")
    
    return Ci, meet_points, all_dist

# ---------------------------
# Calcular costo de una ruta
# ---------------------------
def route_cost(route, all_dist):
    if len(route) < 2:
        return 0
    cost = 0
    for i in range(len(route) - 1):
        if route[i] in all_dist and route[i+1] in all_dist[route[i]]:
            cost += all_dist[route[i]][route[i+1]]
        else:
            all_dist[route[i]] = dijkstra(graph, route[i])
            cost += all_dist[route[i]][route[i+1]]

    return cost

# ---------------------------
# Construcción: Nearest Neighbor aleatorizado
# ---------------------------
def nearest_neighbor_greedy(graph, meet_points, all_dist, depot=0):
    unvisited = set(meet_points)
    unvisited.discard(depot)
    route = [depot]
    current = depot
    
    while unvisited:
        next_node = min(unvisited, key=lambda v: all_dist[current][v])
        print(all_dist[current][next_node])
        if graph[current][next_node] == math.inf:
            path_to_next = dijkstra_with_path(graph, current, next_node)
            route.extend(path_to_next[1:])
        else:
            route.append(next_node)          

        unvisited.remove(next_node)
        current = next_node

    if current != depot:
        if graph[current][depot] == math.inf:
            path_to_next = dijkstra_with_path(graph, current, depot)
            route.extend(path_to_next[1:])
        else:
            route.append(depot)
    
    return route

def nearest_neighbor_farthest_first(meet_points, all_dist, depot=0):
    unvisited = set(meet_points)
    route = [depot]
    
    # Empezar por el punto más lejano del depósito
    if unvisited:
        farthest = max(unvisited, key=lambda v: all_dist[depot][v])
        route.append(farthest)
        unvisited.remove(farthest)
        current = farthest
    
    # Continuar con nearest neighbor normal
    while unvisited:
        next_node = min(unvisited, key=lambda v: all_dist[current][v])
        route.append(next_node)
        unvisited.remove(next_node)
        current = next_node
    
    route.append(depot)
    return route

def savings_algorithm(meet_points, all_dist, depot=0):
    """Algoritmo de Clarke-Wright simplificado"""
    if not meet_points:
        return [depot, depot]
    
    # Calcular ahorros para cada par de puntos
    savings = []
    for i, u in enumerate(meet_points):
        for j, v in enumerate(meet_points[i+1:], i+1):
            s = all_dist[depot][u] + all_dist[depot][v] - all_dist[u][v]
            savings.append((s, u, v))
    
    # Ordenar por ahorro descendente
    savings.sort(reverse=True)
    
    # Construir ruta usando los mejores ahorros
    route = [depot]
    used = set()
    
    for s, u, v in savings:
        if u not in used and v not in used and len(route) == 1:
            route.extend([u, v])
            used.update([u, v])
        elif u not in used and v in route:
            idx = route.index(v)
            route.insert(idx, u)
            used.add(u)
        elif v not in used and u in route:
            idx = route.index(u)
            route.insert(idx + 1, v)
            used.add(v)
    
    # Agregar puntos no incluidos
    for point in meet_points:
        if point not in used:
            route.insert(-1, point)
    
    route.append(depot)
    return route

def nearest_neighbor_random(meet_points, all_dist, depot=0, k=3):
    """Versión original mejorada"""
    unvisited = set(meet_points)
    unvisited.discard(depot)

    route = [depot]
    current = depot

    while unvisited:
        candidates = sorted(unvisited, key=lambda v: all_dist[current][v])
        k_candidates = candidates[:min(k, len(candidates))]
        next_node = random.choice(k_candidates)
        if graph[current][next_node] == math.inf:
            path_to_next = dijkstra_with_path(graph, current, next_node)
            route.extend(path_to_next[1:])
        else:
            route.append(next_node)

        unvisited.remove(next_node)
        current = next_node

    if current != depot:
        if graph[current][depot] == math.inf:
            path_to_next = dijkstra_with_path(graph, current, depot)
            route.extend(path_to_next[1:])
        else:
            route.append(depot)

    return route

def nearest_neighbor_mixed(meet_points, all_dist, depot=0):
    """Versión original mejorada"""
    unvisited = set(meet_points)
    unvisited.discard(depot)

    route = [depot]
    current = depot

    while unvisited:
        k = random.choice([1, 2, 3])
        candidates = sorted(unvisited, key=lambda v: all_dist[current][v])
        k_candidates = candidates[:min(k, len(candidates))]
        next_node = random.choice(k_candidates)
        if graph[current][next_node] == math.inf:
            path_to_next = dijkstra_with_path(graph, current, next_node)
            route.extend(path_to_next[1:])
        else:
            route.append(next_node)

        unvisited.remove(next_node)
        current = next_node

    if current != depot:
        if graph[current][depot] == math.inf:
            path_to_next = dijkstra_with_path(graph, current, depot)
            route.extend(path_to_next[1:])
        else:
            route.append(depot)

    return route

def nearest_neighbor_variants(graph, meet_points, all_dist, depot=0, variant='random'):
    """
    Variantes de nearest neighbor:
    - 'random': aleatorizado (k candidatos)
    - 'greedy': siempre el más cercano
    - 'farthest': más lejano primero (para diversificar)
    - 'savings': basado en algoritmo de ahorros
    """
    if variant == 'random':
        return nearest_neighbor_random(meet_points, all_dist, depot, k=3)
    elif variant == 'greedy':
        return nearest_neighbor_greedy(meet_points, all_dist, depot)
    elif variant == 'farthest':
        return nearest_neighbor_farthest_first(meet_points, all_dist, depot)
    elif variant == 'savings':
        return savings_algorithm(meet_points, all_dist, depot)
    elif variant == 'mixed':
        return nearest_neighbor_mixed(meet_points, all_dist, depot)

    


# ---------------------------
# Múltiples construcciones iniciales
# ---------------------------
def multi_start_construction(graph, workers, depot=0, num_starts=2):
    """Genera múltiples soluciones iniciales y toma la mejor"""
    best_route = None
    best_cost = float('inf')
    best_Ci = None
    best_all_dist = None
    
    #strategies = ['random', 'greedy', 'center', 'mixed']
    #variants = ['random', 'greedy', 'farthest', 'savings']
    
    for _ in range(num_starts):
        #strategy = random.choice(strategies)
        #strategies.remove(strategy)
        strategy = 'greedy'
        #variant = random.choice(variants)
        variant = 'mixed' 

        Ci, meet_points, all_dist = build_initial_solution(
            graph, workers, depot, strategy)


        print(f"Estrategia '{strategy}' generó {len(meet_points)} puntos de encuentro")
        
        route = nearest_neighbor_variants(graph, meet_points, all_dist, depot, variant)
        cost = route_cost(route, all_dist)
        
        if cost < best_cost:
            best_cost = cost
            best_route = route
            best_Ci = Ci
            best_all_dist = all_dist
            print(f"Nueva mejor solución con costo {best_cost} usando estrategia '{strategy}' y variante '{variant}'")
    
    return best_Ci, best_route, best_cost, best_all_dist

# ---------------------------
# 2-opt
# ---------------------------
def two_opt(route, all_dist, best_cost, graph):
    best_route = route[:]
    improved = True
    
    while improved:
        improved = False
        for i in range(1, len(route) - 2):
            for j in range(i + 1, len(route) - 1):
                # Crear nueva ruta intercambiando
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                new_cost = route_cost(new_route, all_dist)
                
                if new_cost < best_cost:
                    path1, path2 = [], []

                    if graph[route[i-1]][route[j]] == math.inf:
                        path1 = dijkstra_with_path(graph, route[i-1], route[j])[1:]  # sacamos el duplicado inicial

                    if graph[route[i]][route[j+1]] == math.inf:
                        path2 = dijkstra_with_path(graph, route[i], route[j+1])[1:]

                    new_route = route[:i] + path1 + route[i:j+1][::-1][1:] + path2 + route[j+1:][1:]
                        
                    best_route = new_route
                    best_cost = new_cost
                    route = new_route
                    improved = True
                    break
            if improved:
                break
    
    return best_route, best_cost

def evaluate_exclusion(graph, workers, Ci, route, cost):
    maxP = 0 
    for i, w in enumerate(workers):
        route_modified = route.copy()
        # 1. Nodos prohibidos para este trabajador
        forbidden_nodes = Ci[i] + [w[0]]
        
        # 2. Construir nueva ruta sin esos nodos
        for node in forbidden_nodes:
            if node in route_modified:
                route_modified.remove(node)

        reduced_graph = remove_nodes(graph, forbidden_nodes)

        new_route = []

        for idx in range(len(route_modified)-1):
            current = route_modified[idx]
            new_route.append(current)
            next_node = route_modified[idx+1]
            if reduced_graph[current][next_node] == math.inf:
                path_to_next = dijkstra_with_path(reduced_graph, current, next_node)
                new_route.extend(path_to_next[1:])
            else:
                new_route.append(next_node) 

        new_cost = sum(reduced_graph[new_route[i]][new_route[i+1]] for i in range(len(new_route)-1))
        
        uncovered = 0
        for (vi, ri) in workers:
            reachable = [v for v, d in all_dist[vi].items() if d <= ri]
            if not any(mp in reachable for mp in new_route):
                uncovered += 1
        
        if uncovered > 0:
            print(f"⚠️ {uncovered} trabajadores no quedaron cubiertos al excluir trabajador {i}")
            P = (cost - new_cost) / uncovered                   
            maxP = max(maxP, P)

    return maxP

# ---------------------------
# Ejemplo de uso integrado
# ---------------------------
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python main.py grafo.csv instancia.csv")
        sys.exit(1)

    grafo_file = sys.argv[1]
    instancia_file = sys.argv[2]

    # iniciar reloj global
    start_time = time.time()

    N, graph = load_graph(grafo_file)
    print(f"Grafo cargado con {N} nodos")

    workers = load_instance(instancia_file)
    print(f"Instancia cargada con {len(workers)} trabajadores")

    # precalcular todas las distancias
    t0 = time.time()

    # fase de construcción: 
    Ci, initial_route, initial_cost, all_dist = multi_start_construction(
        graph, workers, depot=0, num_starts=1)

    t1 = time.time()
    print(f"Tiempo de construcción de solución inicial: {t1 - t0:.2f} segundos")

    print("\nRuta inicial:", initial_route)
    print("Costo inicial:", initial_cost)

    best_route, best_cost = initial_route, initial_cost

    ## fase de mejora: aplicar 2-opt
    for i in range(3):  # aplicar 2-opt dos veces para mejoría adicional
        best_route, best_cost = two_opt(best_route, all_dist, best_cost, graph)
    
    t2 = time.time()
    print(f"Tiempo de mejora con 2-opt: {t2 - t1:.2f} segundos")

    print("\nRuta mejorada (2-opt):", best_route)
    print("Costo mejorado:", best_cost)

    total_time = time.time() - start_time
    print("\nTiempo total: {:.4f} segundos".format(total_time))

    # Evaluar exclusión de trabajadores
    P = evaluate_exclusion(graph, workers, Ci, best_route, best_cost)
    print(f"\nValor máximo P al evaluar exclusión de trabajadores: {P:.4f}")


    # guardar en archivo resultados.txt
    with open("resultados10.txt", "w") as f:
        f.write(" ".join(map(str, best_route)) + "\n")
