# scripts/run_baseline.py
import argparse, time
import pandas as pd
from pathlib import Path
from src.data.simple_loader import load_instance
from src.models.greedy import greedy_construct

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Ruta a .csv (sintético) o .xlsx (real)")
    ap.add_argument("--name", required=True, help="Nombre lógico de la instancia")
    ap.add_argument("--runs", type=int, default=1)
    args = ap.parse_args()

    inst = load_instance(args.file, args.name)

    results = []
    for r in range(args.runs):
        t0 = time.perf_counter()
        route, feas, prof, T, arr = greedy_construct(inst)
        elapsed = time.perf_counter() - t0
        results.append({
            "instance": args.name, "run": r, "feasible": feas,
            "profit": float(prof), "route_time": float(T),
            "stops": max(0, len(route)-2), "compute_sec": float(elapsed)
        })
        print(f"[{args.name}] run={r} feasible={feas} profit={prof:.2f} time={T:.2f} stops={len(route)-2} compute={elapsed:.4f}s")

    Path("experiments/tables").mkdir(parents=True, exist_ok=True)
    out = Path("experiments/tables") / f"baseline_greedy_{args.name}.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"Saved -> {out}")

if __name__ == "__main__":
    main()
