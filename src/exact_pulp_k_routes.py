import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pulp


# -------------------- Carga CSV y matriz de distancias --------------------

def load_csv_with_euclid(path: str):
    df = pd.read_csv(path)
    df = df.rename(columns=str.lower)
    # aliases comunes
    aliases = {"a": "open", "b": "close", "s": "service", "p": "profit"}
    for k, v in aliases.items():
        if k in df.columns:
            df = df.rename(columns={k: v})
    need = ["id", "open", "close", "service", "profit"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"Falta columna '{c}' en CSV (alias a/b/s/p soportados).")

    if "x" not in df.columns:
        df["x"] = 0.0
    if "y" not in df.columns:
        df["y"] = 0.0
    if "is_depot" not in df.columns:
        df["is_depot"] = (df["id"] == 0).astype(int)

    df = df.sort_values("id").reset_index(drop=True)

    # matriz euclídea
    xy = df[["x", "y"]].to_numpy()
    n = len(df)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = float(np.hypot(xy[i, 0] - xy[j, 0], xy[i, 1] - xy[j, 1]))

    # leer k si existe
    k_routes = 1
    if "k" in df.columns:
        try:
            k_routes = int(pd.to_numeric(df["k"]).dropna().iloc[0])
        except Exception:
            k_routes = 1

    return df, D, k_routes


# -------------------- Modelo TOPTW multi-ruta (K rutas) --------------------

def solve_toptw_k_routes(df: pd.DataFrame, D: np.ndarray, K: int, time_limit=60, msg=True):
    n = len(df)
    N = range(n)
    depot = 0
    bigM = 10 ** 5

    mdl = pulp.LpProblem("TOPTW_Exact_K_Routes", pulp.LpMaximize)

    # Variables
    x = pulp.LpVariable.dicts("x", (N, N), lowBound=0, upBound=1, cat=pulp.LpBinary)
    y = pulp.LpVariable.dicts("y", N, lowBound=0, upBound=1, cat=pulp.LpBinary)
    t = pulp.LpVariable.dicts("t", N, lowBound=0, cat=pulp.LpContinuous)

    # No lazos
    for i in N:
        mdl += x[i][i] == 0

    # Ventanas de tiempo condicionadas por visita (para i != depot)
    for i in N:
        if i == depot:
            continue
        open_i = float(df.loc[i, "open"])
        close_i = float(df.loc[i, "close"])
        mdl += t[i] >= open_i - bigM * (1 - y[i])
        mdl += t[i] <= close_i + bigM * (1 - y[i])

    # Flujo en depósito: K salidas y K entradas
    mdl += pulp.lpSum([x[depot][j] for j in N if j != depot]) == K
    mdl += pulp.lpSum([x[j][depot] for j in N if j != depot]) == K

    # Flujo en nodos (i != depot): si visitado, 1 in y 1 out; si no, 0
    for i in N:
        if i == depot:
            continue
        mdl += pulp.lpSum([x[i][j] for j in N if j != i]) == y[i]
        mdl += pulp.lpSum([x[j][i] for j in N if j != i]) == y[i]

    # Coherencia temporal
    # Desde depósito a j: t[j] >= service[0] + D[0,j] - M*(1 - x[0,j])
    serv0 = float(df.loc[depot, "service"]) if "service" in df.columns else 0.0
    for j in N:
        if j == depot:
            continue
        travel = D[depot][j]
        mdl += t[j] >= serv0 + travel - bigM * (1 - x[depot][j])

    # Entre clientes i->j (i!=j, j!=depot)
    for i in N:
        if i == depot:
            continue
        service_i = float(df.loc[i, "service"])
        for j in N:
            if j == depot or j == i:
                continue
            travel = D[i][j]
            mdl += t[j] >= t[i] + service_i + travel - bigM * (1 - x[i][j])

    # Objetivo
    mdl += pulp.lpSum([float(df.loc[i, "profit"]) * y[i] for i in N if i != depot])

    # Solver
    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit)
    status = mdl.solve(solver)
    status_str = pulp.LpStatus[status]
    if msg:
        print("Solver status:", status_str)

    # Reconstrucción de rutas: K rutas desde las K salidas del depósito
    succ = {}
    for i in N:
        for j in N:
            if i != j and pulp.value(x[i][j]) > 0.5:
                succ.setdefault(i, []).append(j)

    # Tomar exactamente los sucesores del depósito como puntos de partida
    starts = [j for j in (succ.get(depot, []) or [])]
    routes = []
    used = set([depot])
    for s in starts:
        route = [depot, s]
        cur = s
        visited_in_route = set([depot, s])
        while True:
            nxts = [v for v in succ.get(cur, [])]
            if not nxts:
                break
            # Si llega al depósito, termina esta ruta
            if depot in nxts:
                route.append(depot)
                break
            # tomar el primero (CBC devuelve una sola salida por nodo no-depot)
            nxt = nxts[0]
            if nxt in visited_in_route:
                # ciclo defensivo
                break
            route.append(nxt)
            visited_in_route.add(nxt)
            cur = nxt
        routes.append(route)

    # Métricas por ruta
    def eval_route(route_idx):
        ttime = 0.0
        prof = 0.0
        for k in range(1, len(route_idx)):
            i = route_idx[k - 1]
            j = route_idx[k]
            ttime += D[i][j]
            if j != depot:
                ttime = max(ttime, float(df.loc[j, "open"]))
                if ttime > float(df.loc[j, "close"]):
                    return False, 0.0, ttime
                ttime += float(df.loc[j, "service"])
                prof += float(df.loc[j, "profit"]) if int(df.loc[j, "is_depot"]) == 0 else 0.0
        return True, prof, ttime

    routes_info = []
    total_profit = 0.0
    total_time = 0.0
    total_stops = 0
    for ridx, rt in enumerate(routes):
        feas, prof, ttime = eval_route(rt)
        stops = max(0, len(rt) - 2)
        routes_info.append(
            {
                "route_id": ridx,
                "feasible": feas,
                "profit": float(prof),
                "route_time": float(ttime),
                "stops": int(stops),
                "route": rt,
            }
        )
        total_profit += prof
        total_time += ttime
        total_stops += stops

    return {
        "status": status_str,
        "objective": float(pulp.value(mdl.objective)) if pulp.value(mdl.objective) is not None else None,
        "routes": routes_info,
        "totals": {
            "profit": float(total_profit),
            "time": float(total_time),
            "stops": int(total_stops),
        },
    }


# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Ruta a CSV (id,x,y,a,b,s,p,...) con columna opcional k")
    ap.add_argument("--name", required=True, help="Nombre de la instancia")
    ap.add_argument("--k", type=int, default=None, help="Número de rutas (si no se especifica, se lee de CSV si existe, por defecto 1)")
    ap.add_argument("--time-limit", type=int, default=60, help="Límite de tiempo del solver (segundos)")
    ap.add_argument("--quiet", action="store_true", help="Silenciar salida del solver")
    args = ap.parse_args()

    p = Path(args.file)
    if p.suffix.lower() != ".csv":
        raise SystemExit("Este exacto k-rutas lee actualmente solo CSV.")

    df, D, k_csv = load_csv_with_euclid(args.file)
    K = args.k if args.k is not None else int(k_csv)

    res = solve_toptw_k_routes(df, D, K, time_limit=args.time_limit, msg=not args.quiet)

    print(f"\n=== RESULTADOS ({args.name}) ===")
    print("Status:", res["status"]) 
    print("Objetivo (max profit):", res["objective"])
    for r in res["routes"]:
        print(
            f"route={r['route_id']} feasible={r['feasible']} profit={r['profit']:.2f} time={r['route_time']:.2f} stops={r['stops']} path={r['route']}"
        )
    print(
        f"TOTAL k={K} profit={res['totals']['profit']:.2f} time_sum={res['totals']['time']:.2f} stops_sum={res['totals']['stops']}"
    )

    # Guardar resultados
    out_dir = Path("experiments/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"exact_k_{args.name}.csv"

    rows = [
        {
            "instance": args.name,
            "k": K,
            "route_id": r["route_id"],
            "feasible": r["feasible"],
            "profit": r["profit"],
            "route_time": r["route_time"],
            "stops": r["stops"],
            "route_nodes": ",".join(map(str, r["route"])) if r["route"] else "",
        }
        for r in res["routes"]
    ]
    pd.DataFrame(rows).to_csv(out_file, index=False)
    print(f"Saved -> {out_file}")


if __name__ == "__main__":
    main()

