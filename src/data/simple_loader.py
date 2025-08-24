import padas as pd
import numpy as np
from pathlib import Path

import pandas as pd
import numpy as np
from pathlib import Path

def _euclid_matrix_from_xy(nodes_df):
    """Construye una matriz de distancias euclídeas usando columnas x,y."""
    xy = nodes_df[['x','y']].to_numpy()
    n  = len(xy)
    D  = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(n):
            if i != j:
                D[i, j] = float(np.hypot(xy[i,0] - xy[j,0], xy[i,1] - xy[j,1]))
    return D

def _to_standard_columns_csv(df):
    """
    Normaliza nombres: s->service, p->profit, a->open, b->close.
    Si faltan x,y, se rellenan con 0. is_depot = (id==0) si no existe.
    """
    # todo en minúsculas para facilitar el mapeo
    df = df.rename(columns={c: c.lower() for c in df.columns})

    mapping = {
        's': 'service', 'service': 'service',
        'p': 'profit',  'profit': 'profit',
        'a': 'open',    'open': 'open',
        'b': 'close',   'close': 'close',
        'x': 'x', 'y': 'y',
        'id': 'id',
        'is_depot': 'is_depot'
    }
    rename = {}
    for k, v in mapping.items():
        if k in df.columns:
            rename[k] = v
    df = df.rename(columns=rename)

    # obligatorias
    required = ['id','open','close','service','profit']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Falta columna obligatoria en CSV: {col}")

    # opcionales
    if 'x' not in df.columns: df['x'] = 0.0
    if 'y' not in df.columns: df['y'] = 0.0
    if 'is_depot' not in df.columns: df['is_depot'] = (df['id'] == 0).astype(int)

    # ordenar por id y dejar solo columnas estándar
    df = df[['id','x','y','open','close','service','profit','is_depot']].copy()
    df = df.sort_values('id').reset_index(drop=True)
    return df

def _read_vector_sheet(xls, sheet_name):
    """Lee la segunda columna de la hoja (suele ser donde vienen los valores)."""
    df = pd.read_excel(xls, sheet_name=sheet_name)
    col_vals = df.columns[1]  # columna 1: valores
    return df[col_vals].to_numpy()

def _read_matrix_DT(xls):
    """Lee la hoja DT y extrae la submatriz numérica (quitando fila/col de etiquetas)."""
    df = pd.read_excel(xls, sheet_name='DT')
    M  = df.iloc[1:, 1:].to_numpy(dtype=float)
    return M

def load_instance(file_path, name):
    """
    Devuelve un diccionario:
    {
      'name': str,
      'nodes': DataFrame con columnas estándar,
      'dist':  matriz NxN (numpy array)
    }
    """
    p = Path(file_path)
    if p.suffix.lower() == '.csv':
        # ---- SINTÉTICO CSV ----
        raw = pd.read_csv(p)
        nodes = _to_standard_columns_csv(raw)
        dist  = _euclid_matrix_from_xy(nodes)  # si tienes dist propia, se puede cambiar luego
        return {'name': name, 'nodes': nodes, 'dist': dist}

    elif p.suffix.lower() in ('.xlsx', '.xls'):
        # ---- REAL XLSX ---- (espera hojas: DT, P, S, A, B)
        xls = pd.ExcelFile(p)
        D = _read_matrix_DT(xls)
        P = _read_vector_sheet(xls, 'P')
        S = _read_vector_sheet(xls, 'S')
        A = _read_vector_sheet(xls, 'A')
        B = _read_vector_sheet(xls, 'B')

        N = len(P)
        # Caso 1: ya viene depósito como primera fila (profit=0 y service=0)
        depot_present = (P[0] == 0) and (S[0] == 0)

        if depot_present:
            ids      = np.arange(N)
            is_depot = (ids == 0).astype(int)
            open_v, close_v, service, profit = A, B, S, P
            dist = D
        else:
            # Insertamos depósito ficticio al inicio
            big = 10**9
            ids      = np.arange(N+1)
            is_depot = (ids == 0).astype(int)
            open_v   = np.concatenate([[0], A])
            close_v  = np.concatenate([[big], B])
            service  = np.concatenate([[0], S])
            profit   = np.concatenate([[0], P])
            dist = np.zeros((N+1, N+1), dtype=float)
            dist[1:, 1:] = D  # dejamos 0 para viajes desde/hacia depósito (mejorable)

        nodes = pd.DataFrame({
            'id': ids, 'x': 0.0, 'y': 0.0,
            'open': open_v, 'close': close_v,
            'service': service, 'profit': profit,
            'is_depot': is_depot
        }).sort_values('id').reset_index(drop=True)

        return {'name': name, 'nodes': nodes, 'dist': dist}

    else:
        raise ValueError(f"Extensión no soportada: {p.suffix}")
