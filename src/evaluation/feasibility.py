def evaluate_route(route_idx, nodes_df, dist):
    """
    route_idx: lista de índices (ej. [0, 3, 5, 0]) con depósito al inicio y final.
    Devuelve: (feasible, total_profit, total_time, arrivals_list)
    """
    t = 0.0          # reloj
    profit = 0.0     # utilidad acumulada
    arrivals = []    # tiempos de llegada a cada visita (útil para diagnóstico)
    last = route_idx[0]
    feasible = True

    for k in range(1, len(route_idx)):
        v = route_idx[k]
        # 1) viajar
        t += dist[last, v]

        # 2) esperar si llegas antes de la apertura
        open_v = nodes_df.loc[v, 'open']
        close_v = nodes_df.loc[v, 'close']
        serv_v = nodes_df.loc[v, 'service']
        prof_v = nodes_df.loc[v, 'profit']

        if t < open_v:
            t = open_v

        arrivals.append(t)

        # 3) ventana de tiempo
        if t > close_v:
            feasible = False
            break

        # 4) prestar servicio
        t += serv_v

        # 5) sumar profit si no es depósito
        if int(nodes_df.loc[v, 'is_depot']) == 0:
            profit += prof_v

        last = v

    return feasible, profit, t, arrivals