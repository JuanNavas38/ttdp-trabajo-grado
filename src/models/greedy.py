# src/models/greedy.py
import numpy as np
from src.evaluation.feasibility import evaluate_route

def greedy_construct(instance, start_idx=0, return_to_depot=True, tie_break='profit_density'):
    nodes = instance['nodes']
    dist  = instance['dist']
    N = len(nodes)

    # candidatos iniciales: todos menos el depósito
    unvisited = [i for i in range(N) if i != start_idx and int(nodes.loc[i, 'is_depot']) == 0]
    route = [start_idx]

    while unvisited:
        best_v = None
        best_key = (-np.inf, )

        for v in unvisited:
            # ruta de prueba: ruta + v + regreso (si aplica) para medir factibilidad completa
            trial = route + [v]
            if return_to_depot:
                trial = trial + [start_idx]

            feas, prof, T, _ = evaluate_route(trial, nodes, dist)
            if not feas:
                continue

            # criterio simple: densidad de beneficio sobre el T total de trial
            score_v = nodes.loc[v, 'profit']
            if tie_break == 'profit_density':
                val = score_v / max(1e-6, T)
            else:
                val = score_v

            if val > best_key[0]:
                best_key = (val, prof, -T)
                best_v = v

        if best_v is None:
            # no hay ningún candidato que quepa → paramos
            break

        # insertar antes del regreso (si se regresa)
        if return_to_depot and route[-1] == start_idx and len(route) > 0:
            route.insert(len(route), best_v)  # [0, v1] ... luego se añadirá 0 al final si falta
        else:
            route.append(best_v)

        unvisited.remove(best_v)

    # cerrar en depósito si hace falta
    if return_to_depot and route[-1] != start_idx:
        route.append(start_idx)

    feas, prof, T, arr = evaluate_route(route, nodes, dist)
    return route, feas, prof, T, arr
