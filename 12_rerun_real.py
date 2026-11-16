# Reference implementation for the LBRD-PPFL manuscript.

import argparse, importlib.util, json, sys, time
from pathlib import Path
try:
    import h5py
    _HAS_H5PY = True
except ImportError:
    h5py = None
    _HAS_H5PY = False
import numpy as np
import torch
PROJ = Path(__file__).resolve().parent
if not (PROJ / '10_rerun_unified.py').exists():
    PROJ = Path('E:\\code\\biology1.03')
sys.path.insert(0, str(PROJ))
_spec = importlib.util.spec_from_file_location('u10', str(PROJ / '10_rerun_unified.py'))
u10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(u10)
REAL_DIR = PROJ / 'data' / 'processed_real'
_REAL_DIR = REAL_DIR
PREFIX = {'ECG': 'ecg', 'PPG': 'ppg', 'EEG': 'eeg'}
_LBRD_MODE = 'augment'
METHODS = ['fedavg', 'dpfedavg', 'fedprox', 'dpfedcgan', 'lbrd', 'lbrd_dp']

def _match_rows(host: np.ndarray, sub: np.ndarray) -> np.ndarray:
    h = host.reshape(host.shape[0], -1).astype(np.float32)
    s = sub.reshape(sub.shape[0], -1).astype(np.float32)
    index: dict = {}
    for i, row in enumerate(h):
        index.setdefault(row.tobytes(), []).append(i)
    hits = []
    for row in s:
        cand = index.get(row.tobytes())
        if not cand:
            raise ValueError(f'{sub.shape}')
        hits.append(cand.pop(0))
    return np.asarray(hits, dtype=np.int64)

def _npz_path(robust_cache: str):
    p = Path(robust_cache)
    if p.suffix.lower() in ('.hdf5', '.h5') and (not _HAS_H5PY):
        npz = p.with_suffix('.npz')
        if npz.exists():
            return npz
    return p

def _read_robust_field(robust_cache: str, field: str):
    d = {}
    p = _npz_path(robust_cache)
    if p.suffix.lower() == '.npz':
        z = np.load(p)
        for cid in z['_clients'].tolist():
            d[int(cid)] = (np.asarray(z[f'{field}_data_{cid}'], dtype=np.float32), np.asarray(z[f'{field}_labels_{cid}'], dtype=np.int64))
        return d
    with h5py.File(p, 'r') as f:
        for k in f.keys():
            if not k.startswith('client'):
                continue
            cid = int(k.split('_')[1])
            d[cid] = (np.asarray(f[k][f'{field}_data'][:], dtype=np.float32), np.asarray(f[k][f'{field}_labels'][:], dtype=np.int64))
    return d

def _load_robust(robust_cache: str):
    return _read_robust_field(robust_cache, 'robust')

def _load_robust_raw(robust_cache: str):
    return _read_robust_field(robust_cache, 'raw')

def load_real(prefix: str, cfg, use_robust=False, protocol='noleak', robust_cache=None, lbrd_mode: str=None):
    mode = lbrd_mode or _LBRD_MODE
    base = _REAL_DIR
    rob = _load_robust(robust_cache) if use_robust and robust_cache else {}
    robsrc = _load_robust_raw(robust_cache) if use_robust and robust_cache and (mode == 'replace') else {}
    clients = []
    for i in range(cfg.n_clients):
        tr_x = np.load(base / f'{prefix}_client{i}_train_X.npy').astype(np.float32)
        tr_y = np.load(base / f'{prefix}_client{i}_train_y.npy').astype(np.int64)
        te_x = np.load(base / f'{prefix}_client{i}_test_X.npy').astype(np.float32)
        te_y = np.load(base / f'{prefix}_client{i}_test_y.npy').astype(np.int64)
        tr_x = torch.FloatTensor(tr_x)
        te_x = torch.FloatTensor(te_x)
        if tr_x.dim() == 2:
            tr_x = tr_x.unsqueeze(1)
        if te_x.dim() == 2:
            te_x = te_x.unsqueeze(1)
        tr_y = torch.LongTensor(tr_y)
        kept = 0
        rep_x = rep_y = None
        if use_robust and i in rob:
            rx = torch.FloatTensor(rob[i][0])
            if rx.dim() == 2:
                rx = rx.unsqueeze(1)
            ry = torch.LongTensor(rob[i][1])
            if mode == 'replace':
                si = _match_rows(tr_x.cpu().numpy(), robsrc[i][0])
                rep_x, rep_y = (tr_x[si], tr_y[si])
                keep = np.ones(tr_x.shape[0], dtype=bool)
                keep[si] = False
                tr_x = torch.cat([tr_x[keep], rx], 0)
                tr_y = torch.cat([tr_y[keep], ry], 0)
            else:
                tr_x = torch.cat([tr_x, rx], 0)
                tr_y = torch.cat([tr_y, ry], 0)
            kept = rx.shape[0]
        clients.append({'train_data': tr_x, 'train_labels': tr_y, 'test_data': te_x, 'test_labels': torch.LongTensor(te_y), 'train_size': tr_x.shape[0], 'n_robust_kept': kept, 'lbrd_mode': mode, 'replaced_data': rep_x, 'replaced_labels': rep_y})
    return clients

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(PREFIX))
    ap.add_argument('--method', required=True, choices=METHODS)
    ap.add_argument('--rounds', type=int, default=100)
    ap.add_argument('--local_epochs', type=int, default=3)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--lr', type=float, default=0.01)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epsilon', type=float, default=2.0)
    ap.add_argument('--delta', type=float, default=1e-05)
    ap.add_argument('--sigma', type=float, nargs='+', default=None)
    ap.add_argument('--clip_norm', type=float, default=1.0)
    ap.add_argument('--tag', default=None)
    ap.add_argument('--real_dir', default=None, help='')
    ap.add_argument('--robust_cache', default=None, help='')
    ap.add_argument('--lbrd_mode', choices=['augment', 'replace'], default='augment', help='')
    ap.add_argument('--out', default=str(PROJ / 'experiments' / 'rerun_real'))
    a = ap.parse_args()
    global _REAL_DIR, _LBRD_MODE
    if a.real_dir:
        _REAL_DIR = Path(a.real_dir)
    _LBRD_MODE = a.lbrd_mode
    prefix = PREFIX[a.dataset]
    u10.DATASETS[a.dataset] = dict(cache='<processed_real>', input_channels=1, seq_length=1000, num_classes=5, hidden_dims=[32, 64, 128])
    cfg = u10.Cfg(a.dataset, num_rounds=a.rounds, local_epochs=a.local_epochs, batch_size=a.batch_size, lr=a.lr, seed=a.seed, epsilon=a.epsilon, delta=a.delta, clip_norm=a.clip_norm)
    u10.load_cache = lambda path, c, ur, pr, rc=None: load_real(prefix, c, ur, pr, rc, _LBRD_MODE)
    res_dir = Path(a.out)
    res_dir.mkdir(parents=True, exist_ok=True)
    sigmas = a.sigma if a.sigma else [None]
    for s in sigmas:
        tag = a.tag or f'REAL_{a.dataset}_{a.method}' + ('_rep' if _LBRD_MODE == 'replace' else '') + (f'_sig{s:g}' if s is not None else '') + f'_s{a.seed}'
        out = u10.run(cfg, a.method, a.robust_cache, s, tag)
        (res_dir / f'{tag}.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
        print(f"[{tag}] saved -> {res_dir / (tag + '.json')}")
if __name__ == '__main__':
    main()