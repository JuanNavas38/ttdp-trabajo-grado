# exact_pulp.py
# Uso:
#   python exact_pulp.py data/synthetic/hptoptw-j11a.csv --time-limit 30
# Requiere: pip install pulp (y opcionalmente: pip install coin-or-cbc)

import argparse
import pandas as pd
import numpy as np
import pulp

def load_csv(path):
    df = pd.read_csv(path)
    df = df.rename(columns=str.lower)
    # Renombrar alias comunes
    aliases = {'a':'open', 'b':'close', 's':'service', 'p':'profit'}
    for k,v in aliases.items():
        if k in df.columns: df = df.rename(columns={k:v})
    # Verificar columnas mínimas
    need = ['id','open','close','service','profit']
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Falta columna '{c}' en CSV (alias a/b/s/p soportados).")
    # x,y opcionales (para dist euclídea)
    if 'x' not in df.columns: df['x'] = 0.0
    if 'y' not in df.columns: df['y'] = 0.0
    if 'is_depot' not in df.columns: df['is_depot'] = (df['id']==0).astype(int)
    df = df.sort_values('id').reset_index(drop=True)
    # Matriz de viaje euclídea
    xy = df[['x','y']].to_numpy()
    n  = len(df)
    D  = np.zeros((n,n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i,j] = float(np.hypot(xy[i,0]-xy[j,0], xy[i,1]-xy[j,1]))
    return df, D

def solve_toptw_single_route(df, D, time_limit=30, msg=True):
    """
    TOPTW/TTDP con 1 ruta que inicia/termina en depósito (id=0).
    Variables:
      x[i,j] ∈ {0,1}  -> tomar arco i->j
      y[i]   ∈ {0,1}  -> visitar nodo i
      t[i]   ≥ 0      -> tiempo de llegada a i
    Objetivo: maximizar sum(profit[i] * y[i]) (i != depot)
    """
    n = len(df)
    N = range(n)
    depot = 0
    bigM = 10**5  # Ajusta si tus tiempos están más grandes/pequeños

    # Modelo
    mdl = pulp.LpProblem("TOPTW_Exact_CBC", pulp.LpMaximize)

    # Variables
    x = pulp.LpVariable.dicts("x", (N,N), lowBound=0, upBound=1, cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts("y", N, lowBound=0, upBound=1, cat=pulp.LpBinary)
    t = pulp.LpVariable.dicts("t", N, lowBound=0, cat=pulp.LpContinuous)

    # No lazos
    for i in N:
        mdl += x[i][i] == 0

    # Ventanas de tiempo
    for i in N:
        mdl += t[i] >= float(df.loc[i,'open'])
        mdl += t[i] <= float(df.loc[i,'close'])

    # Flujo en depósito: 1 salida y 1 entrada
    mdl += pulp.lpSum([x[depot][j] for j in N if j != depot]) == 1
    mdl += pulp.lpSum([x[j][depot] for j in N if j != depot]) == 1

    # Flujo en nodos (i != depot): si visitado, 1 in y 1 out; si no, 0
    for i in N:
        if i == depot: 
            continue
        mdl += pulp.lpSum([x[i][j] for j in N if j != i]) == y[i]
        mdl += pulp.lpSum([x[j][i] for j in N if j != i]) == y[i]

    # Coherencia temporal (arcos activados)
    for i in N:
        for j in N:
            if i == j: 
                continue
            travel = D[i][j]
            service_i = float(df.loc[i,'service'])
            # t[j] >= t[i] + service[i] + travel[i,j] - bigM*(1 - x[i,j])
            mdl += t[j] >= t[i] + service_i + travel - bigM*(1 - x[i][j])

    # Fijar depósito como visitado y tiempo inicial (opcional)
    mdl += y[depot] == 1
    # Si tu depósito tiene ventana abierta en 0, esto basta.

    # Objetivo: maximizar profits (no contemos depósito)
    mdl += pulp.lpSum([float(df.loc[i,'profit']) * y[i] for i in N if i != depot])

    # Resolver con CBC
    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit)
    result_status = mdl.solve(solver)

    status_str = pulp.LpStatus[result_status]
    if msg:
        print("Solver status:", status_str)

    if status_str not in ("Optimal", "Not Solved", "Infeasible", "Unbounded", "Undefined"):
        print("Estado solver:", status_str)

    # Reconstruir tour simple desde depósito
    succ = {}
    for i in N:
        for j in N:
            if i != j and pulp.value(x[i][j]) > 0.5:
                succ[i] = j

    route = [depot]
    seen = set([depot])
    # sigue sucesores hasta volver al depósito o detenerse
    while True:
        nxt = succ.get(route[-1], None)
        if nxt is None or nxt == depot:
            break
        if nxt in seen:
            # ciclo detectado (poco probable con TW), rompo
            break
        route.append(nxt)
        seen.add(nxt)
    route.append(depot)

    profit = sum(float(df.loc[i,'profit']) for i in route if i != depot)
    # tiempo total aproximado: t del último visitado + service + regreso al depot
    if len(route) > 2:
        last = route[-2]
        total_time = pulp.value(t[last]) + float(df.loc[last,'service']) + D[last][depot]
    else:
        total_time = 0.0

    return {
        "status": status_str,
        "objective": pulp.value(mdl.objective),
        "route": route,
        "profit": profit,
        "total_time": total_time,
        "y": {i: int(round(pulp.value(y[i]))) for i in N},
        "t": {i: float(pulp.value(t[i])) for i in N}
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", help="Ruta a CSV (id,x,y,a,b,s,p,...)")
    ap.add_argument("--time-limit", type=int, default=30, help="Límite de tiempo del solver (segundos)")
    args = ap.parse_args()

    df, D = load_csv(args.csv_path)
    res = solve_toptw_single_route(df, D, time_limit=args.time_limit, msg=True)

    print("\n=== RESULTADOS ===")
    print("Status:", res["status"])
    print("Objetivo (max profit):", res["objective"])
    print("Ruta:", res["route"])
    print("Profit ruta:", res["profit"])
    print("Tiempo total aprox:", round(res["total_time"], 2))
    print("Visitados y[i]=1:", [i for i,v in res["y"].items() if v==1])
    # Imprime tiempos de llegada útiles para debugging
    print("t[i] (llegadas):", {k: round(v,2) for k,v in res["t"].items()})

if __name__ == "__main__":
    main()
