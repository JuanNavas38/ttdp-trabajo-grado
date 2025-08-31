# baseline_greedy.py
# Uso: python baseline_greedy.py --file data/synthetic/ttdp-small/hptoptw-j11a.csv --name hptoptw-j11a --runs 3
#      python baseline_greedy.py --file data/real/toptw-jclass-a.xlsx --name toptw-jClass-a --runs 3

import argparse, time
from pathlib import Path
import numpy as np
import pandas as pd

# -------------------- Loader sencillo --------------------

def _euclid_matrix_from_xy(nodes_df):
    xy = nodes_df[['x','y']].to_numpy()
    n  = len(xy)
    D  = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = float(np.hypot(xy[i,0] - xy[j,0], xy[i,1] - xy[j,1]))
    return D

def _to_standard_columns_csv(df):
    df = df.rename(columns={c: c.lower() for c in df.columns})
    mapping = {
        's':'service','service':'service',
        'p':'profit','profit':'profit',
        'a':'open','open':'open',
        'b':'close','close':'close',
        'x':'x','y':'y',
        'id':'id','is_depot':'is_depot'
    }
    rename = {}
    for k,v in mapping.items():
        if k in df.columns:
            rename[k] = v
    df = df.rename(columns=rename)
    required = ['id','open','close','service','profit']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Falta columna '{col}' en el CSV (admite s/p/a/b como alias).")
    if 'x' not in df.columns: df['x'] = 0.0
    if 'y' not in df.columns: df['y'] = 0.0
    if 'is_depot' not in df.columns: df['is_depot'] = (df['id']==0).astype(int)
    df = df[['id','x','y','open','close','service','profit','is_depot']].sort_values('id').reset_index(drop=True)
    return df

def _read_vector_sheet(xls, sheet_name):
    df = pd.read_excel(xls, sheet_name=sheet_name)
    col_vals = df.columns[1]  # normalmente la 2da columna
    return df[col_vals].to_numpy()

def _read_matrix_DT(xls):
    df = pd.read_excel(xls, sheet_name='DT')
    return df.iloc[1:,1:].to_numpy(dtype=float)

def load_instance(file_path, name):
    p = Path(file_path)
    if p.suffix.lower() == '.csv':
        raw = pd.read_csv(p)
        nodes = _to_standard_columns_csv(raw)
        dist  = _euclid_matrix_from_xy(nodes)
        return {'name': name, 'nodes': nodes, 'dist': dist}

    elif p.suffix.lower() in ('.xlsx', '.xls'):
        xls = pd.ExcelFile(p)
        D = _read_matrix_DT(xls)
        P = _read_vector_sheet(xls,'P')
        S = _read_vector_sheet(xls,'S')
        A = _read_vector_sheet(xls,'A')
        B = _read_vector_sheet(xls,'B')
        N = len(P)
        depot_present = (P[0]==0) and (S[0]==0)

        if depot_present:
            ids = np.arange(N)
            is_dep = (ids==0).astype(int)
            nodes = pd.DataFrame({
                'id': ids, 'x': 0.0, 'y': 0.0,
                'open': A, 'close': B, 'service': S, 'profit': P,
                'is_depot': is_dep
            }).sort_values('id').reset_index(drop=True)
            dist = D
        else:
            big = 10**9
            ids = np.arange(N+1)
            is_dep = (ids==0).astype(int)
            nodes = pd.DataFrame({
                'id': ids, 'x': 0.0, 'y': 0.0,
                'open': np.concatenate([[0], A]),
                'close': np.concatenate([[big], B]),
                'service': np.concatenate([[0], S]),
                'profit': np.concatenate([[0], P]),
                'is_depot': is_dep
            }).sort_values('id').reset_index(drop=True)
            dist = np.zeros((N+1,N+1), dtype=float)
            dist[1:,1:] = D  # viajes desde/hacia depósito = 0 (suficiente para baseline)

        return {'name': name, 'nodes': nodes, 'dist': dist}

    else:
        raise ValueError(f"Extensión no soportada: {p.suffix}")

# -------------------- Evaluador --------------------

def evaluate_route(route_idx, nodes_df, dist):
    """
    route_idx: lista como [0, v1, v2, ..., 0]
    Retorna: (feasible, total_profit, total_time, arrivals_list)
    """
    t = 0.0
    profit = 0.0
    arrivals = []
    last = route_idx[0]
    feasible = True

    for k in range(1, len(route_idx)):
        v = route_idx[k]
        t += dist[last, v]  # viaje

        open_v  = nodes_df.loc[v,'open']
        close_v = nodes_df.loc[v,'close']
        serv_v  = nodes_df.loc[v,'service']
        prof_v  = nodes_df.loc[v,'profit']

        if t < open_v:  # espera
            t = open_v

        arrivals.append(t)

        if t > close_v:  # violación de ventana
            feasible = False
            break

        t += serv_v  # servicio

        if int(nodes_df.loc[v,'is_depot']) == 0:
            profit += prof_v

        last = v

    return feasible, profit, t, arrivals

# -------------------- Greedy (baseline) --------------------

def greedy_construct(instance, start_idx=0, return_to_depot=True):
    nodes = instance['nodes']
    dist  = instance['dist']
    N = len(nodes)

    unvisited = [i for i in range(N) if i != start_idx and int(nodes.loc[i,'is_depot'])==0]
    route = [start_idx]

    while unvisited:
        best_v = None
        best_key = (-np.inf, )

        for v in unvisited:
            trial = route + [v]
            if return_to_depot:
                trial = trial + [start_idx]
            feas, prof, T, _ = evaluate_route(trial, nodes, dist)
            if not feas:
                continue
            # criterio simple: densidad de beneficio
            score_v = nodes.loc[v,'profit']
            val = score_v / max(1e-6, T)
            if val > best_key[0]:
                best_key = (val, prof, -T)
                best_v = v

        if best_v is None:
            break

        if return_to_depot and route[-1]==start_idx:
            route.insert(len(route), best_v)
        else:
            route.append(best_v)
        unvisited.remove(best_v)

    if return_to_depot and route[-1] != start_idx:
        route.append(start_idx)

    return evaluate_route(route, nodes, dist), route

# -------------------- Main (CLI) --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Ruta a .csv (sintético) o .xlsx (real)")
    ap.add_argument("--name", required=True, help="Nombre de la instancia")
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()

    inst = load_instance(args.file, args.name)

    rows = []
    for r in range(args.runs):
        t0 = time.perf_counter()
        (feas, prof, T, arr), route = greedy_construct(inst)
        elapsed = time.perf_counter() - t0

        stops = max(0, len(route)-2)
        print(f"[{args.name}] run={r} feasible={feas} profit={prof:.2f} time={T:.2f} stops={stops} compute={elapsed:.4f}s")

        rows.append({
            "instance": args.name, "run": r, "feasible": feas,
            "profit": float(prof), "route_time": float(T),
            "stops": int(stops), "compute_sec": float(elapsed)
        })

    # guardar resultados
    out_dir = Path("experiments/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"baseline_greedy_{args.name}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved -> {out}")

if __name__ == "__main__":
    main()
