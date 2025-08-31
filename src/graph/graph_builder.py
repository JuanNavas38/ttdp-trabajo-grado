import argparse
from pathlib import Path
import math
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx


def _euclid_matrix_from_xy(nodes_df: pd.DataFrame) -> np.ndarray:
    xy = nodes_df[["x", "y"]].to_numpy()
    n = len(xy)
    D = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = float(math.hypot(xy[i, 0] - xy[j, 0], xy[i, 1] - xy[j, 1]))
    return D


def _to_standard_columns_csv(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={c: c.lower() for c in df.columns})
    mapping = {
        "s": "service",
        "service": "service",
        "p": "profit",
        "profit": "profit",
        "a": "open",
        "open": "open",
        "b": "close",
        "close": "close",
        "x": "x",
        "y": "y",
        "id": "id",
        "is_depot": "is_depot",
    }
    rename = {}
    for k, v in mapping.items():
        if k in df.columns:
            rename[k] = v
    df = df.rename(columns=rename)
    # columnas requeridas
    required = ["id", "open", "close", "service", "profit"]
    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"Falta columna '{col}' en el CSV (admite s/p/a/b como alias)."
            )
    # opcionales
    if "x" not in df.columns:
        df["x"] = 0.0
    if "y" not in df.columns:
        df["y"] = 0.0
    if "is_depot" not in df.columns:
        df["is_depot"] = (df["id"] == 0).astype(int)
    # ordenar
    df = (
        df[["id", "x", "y", "open", "close", "service", "profit", "is_depot"]]
        .sort_values("id")
        .reset_index(drop=True)
    )
    return df


def load_synthetic_instance_csv(path: str, name: Optional[str] = None):
    """Carga una instancia sintética CSV en el formato esperado y devuelve
    nodes, dist y metadatos globales (k, i, u, l, instance).
    """
    p = Path(path)
    raw = pd.read_csv(p)
    nodes = _to_standard_columns_csv(raw)
    dist = _euclid_matrix_from_xy(nodes)
    meta = {"instance": name or p.stem, "source": "synthetic"}
    for key in ("k", "i", "u", "l"):
        if key in raw.columns:
            try:
                meta[key] = int(pd.to_numeric(raw[key]).dropna().iloc[0])
            except Exception:
                pass
    # consistencia de i
    meta["i_actual"] = len(nodes)
    return {"name": meta["instance"], "nodes": nodes, "dist": dist, "meta": meta}


def edge_temporal_feasibility(i: int, j: int, nodes: pd.DataFrame, dist: np.ndarray) -> Tuple[bool, float]:
    """Aproxima la factibilidad temporal i->j y devuelve (feasible, slack).
    Usa salida temprana de i como open_i + service_i.
    """
    open_i = float(nodes.loc[i, "open"]) if i < len(nodes) else 0.0
    serv_i = float(nodes.loc[i, "service"]) if i < len(nodes) else 0.0
    open_j = float(nodes.loc[j, "open"]) if j < len(nodes) else 0.0
    close_j = float(nodes.loc[j, "close"]) if j < len(nodes) else 0.0
    travel = float(dist[i, j])
    depart_i = open_i + serv_i
    arrival_j = max(open_j, depart_i + travel)
    feasible = arrival_j <= close_j
    slack = (close_j - arrival_j) if feasible else float("-inf")
    return feasible, float(slack)


def build_graph(
    nodes: pd.DataFrame,
    dist: np.ndarray,
    k: Optional[int] = None,
    tw_filter: bool = True,
    alpha: float = 0.1,
    directed: bool = False,
) -> nx.Graph:
    """Construye un grafo POI-POI con atributos en nodos y aristas.
    - k: si se especifica, aplica k-NN por nodo para reducir densidad.
    - tw_filter: si True, filtra aristas no factibles por ventanas de tiempo (aprox. local).
    - alpha: factor de decaimiento para el peso de arista: weight = exp(-alpha * dist).
    - directed: True para DiGraph, False para grafo no dirigido.
    """
    G = nx.DiGraph() if directed else nx.Graph()
    n = len(nodes)

    # añadir nodos
    for i in range(n):
        G.add_node(
            int(nodes.loc[i, "id"]),
            x=float(nodes.loc[i, "x"]),
            y=float(nodes.loc[i, "y"]),
            open=float(nodes.loc[i, "open"]),
            close=float(nodes.loc[i, "close"]),
            service=float(nodes.loc[i, "service"]),
            profit=float(nodes.loc[i, "profit"]),
            is_depot=int(nodes.loc[i, "is_depot"]),
        )

    # estrategia k-NN
    idxs = list(range(n))
    if directed:
        for i in idxs:
            # ordenar candidatos por distancia (excluye i)
            order = sorted([j for j in idxs if j != i], key=lambda j: dist[i, j])
            cand_js = order[:k] if isinstance(k, int) and k > 0 else order
            for j in cand_js:
                feas, slack = edge_temporal_feasibility(i, j, nodes, dist)
                if tw_filter and not feas:
                    continue
                w = math.exp(-alpha * float(dist[i, j]))
                G.add_edge(int(nodes.loc[i, "id"]), int(nodes.loc[j, "id"]),
                           dist=float(dist[i, j]), weight=float(w), tw_feasible=bool(feas), tw_slack=float(slack))
    else:
        # no dirigido: generar candidatos por pares a partir de k-NN simétrico
        pairs = set()
        topk = {}
        for i in idxs:
            order = sorted([j for j in idxs if j != i], key=lambda j: dist[i, j])
            topk[i] = set(order[:k] if isinstance(k, int) and k > 0 else order)
        for i in idxs:
            for j in topk[i]:
                a, b = (i, j) if i < j else (j, i)
                if (a, b) in pairs:
                    continue
                # aplicar filtro TW: al menos una dirección factible
                feas_ij, slack_ij = edge_temporal_feasibility(a, b, nodes, dist)
                feas_ji, slack_ji = edge_temporal_feasibility(b, a, nodes, dist)
                if tw_filter and not (feas_ij or feas_ji):
                    continue
                w = math.exp(-alpha * float(dist[a, b]))
                # preferir slack de la dirección más laxa
                slack = max(slack_ij, slack_ji)
                pairs.add((a, b))
                G.add_edge(int(nodes.loc[a, "id"]), int(nodes.loc[b, "id"]),
                           dist=float(dist[a, b]), weight=float(w), tw_feasible=bool(feas_ij or feas_ji), tw_slack=float(slack))
    return G


def save_graph_artifacts(G: nx.Graph, name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # GraphML
    graphml_path = out_dir / f"{name}.graphml"
    try:
        nx.write_graphml(G, graphml_path)
        print(f"Saved GraphML -> {graphml_path}")
    except Exception as e:
        print(f"Warning: GraphML not saved ({e})")
    # Edgelist CSV
    rows = []
    for u, v, d in G.edges(data=True):
        rows.append({
            "u": u, "v": v,
            "dist": float(d.get("dist", 0.0)),
            "weight": float(d.get("weight", 1.0)),
            "tw_feasible": bool(d.get("tw_feasible", True)),
            "tw_slack": float(d.get("tw_slack", 0.0)),
        })
    df_e = pd.DataFrame(rows)
    edgelist_path = out_dir / f"{name}_edgelist.csv"
    df_e.to_csv(edgelist_path, index=False)
    print(f"Saved edgelist -> {edgelist_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Ruta a CSV sintético (id,x,y,a,b,s,p,...) con metadatos k,i,u,l")
    ap.add_argument("--name", required=False, help="Nombre de la instancia (por defecto, nombre del archivo)")
    ap.add_argument("--knn", type=int, default=None, help="k-NN por nodo (si no se especifica, grafo denso)")
    ap.add_argument("--alpha", type=float, default=0.1, help="Decaimiento para peso de arista: exp(-alpha*dist)")
    ap.add_argument("--no-tw-filter", action="store_true", help="No filtrar por compatibilidad de ventanas de tiempo")
    ap.add_argument("--directed", action="store_true", help="Construir DiGraph (dirigido)")
    args = ap.parse_args()

    inst = load_synthetic_instance_csv(args.file, name=args.name)
    name = inst["name"]
    nodes = inst["nodes"]
    dist = inst["dist"]
    meta = inst.get("meta", {})

    G = build_graph(
        nodes,
        dist,
        k=args.knn,
        tw_filter=not args.no_tw_filter,
        alpha=args.alpha,
        directed=args.directed,
    )

    # Anotar metadatos del grafo
    G.graph.update({
        "name": name,
        **{k: v for k, v in meta.items()}
    })

    # Guardar artefactos
    flags = [
        f"knn{args.knn if args.knn is not None else 'all'}",
        f"alpha{args.alpha}",
        f"tw{0 if args.no_tw_filter else 1}",
        "dir" if args.directed else "undir",
    ]
    out_name = f"{name}_" + "_".join(flags)
    out_dir = Path("experiments/graphs")
    save_graph_artifacts(G, out_name, out_dir)


if __name__ == "__main__":
    main()

