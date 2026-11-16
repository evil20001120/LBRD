# Reference implementation for the LBRD-PPFL manuscript.

import argparse
from pathlib import Path
import numpy as np
AAMI = {}
for s in ['N', 'L', 'R', 'e', 'j']:
    AAMI[s] = 0
for s in ['A', 'a', 'J', 'S']:
    AAMI[s] = 1
for s in ['V', 'E']:
    AAMI[s] = 2
AAMI['F'] = 3
AAMI['Q'] = 4
AAMI['?'] = 4
BEAT_SYMBOLS = set(AAMI)
RECORD_TO_CLASS = {'100': 0, '101': 0, '102': 0, '103': 0, '104': 0, '105': 0, '106': 1, '107': 1, '108': 1, '109': 1, '111': 1, '112': 1, '113': 2, '114': 2, '115': 2, '116': 2, '117': 2, '118': 2, '119': 3, '121': 3, '122': 3, '123': 3, '124': 3, '200': 3, '201': 4, '202': 4, '203': 4, '205': 4, '207': 4, '208': 4, '209': 4, '210': 4}

def map_label(sym: str) -> int:
    return AAMI.get(sym, 4)

def extract_record(rec_path: Path, win: int=1000, fs: int=360):
    import wfdb
    rec = wfdb.rdrecord(str(rec_path))
    ann = wfdb.rdann(str(rec_path), 'atr')
    sig = rec.p_signal[:, 0].astype(np.float64)
    half = win // 2
    segs, labels = ([], [])
    for s, sym in zip(ann.sample, ann.symbol):
        if sym not in AAMI:
            continue
        a, b = (s - half, s + half)
        if a < 0 or b > len(sig):
            continue
        seg = sig[a:b]
        if len(seg) != win:
            continue
        if len(seg) != 1000:
            seg = np.interp(np.linspace(0, 1, 1000), np.linspace(0, 1, len(seg)), seg)
        seg = (seg - seg.mean()) / (seg.std() + 1e-08)
        segs.append(seg.astype(np.float32))
        labels.append(map_label(sym))
    return (np.array(segs), np.array(labels))

def extract_record_level(rec_path: Path, fs: int=360, win_s: float=10.0, seg_s: float=1.0, step_s: float=0.5, std_thr: float=0.1, out_len: int=1000):
    import wfdb
    rec = wfdb.rdrecord(str(rec_path), channels=[0], sampto=int(fs * win_s))
    sig = rec.p_signal.flatten().astype(np.float64)
    seg_n, step_n = (int(fs * seg_s), int(fs * step_s))
    lab = RECORD_TO_CLASS.get(rec_path.stem, 0)
    segs, labels = ([], [])
    for i in range(0, len(sig) - seg_n, step_n):
        seg1k = np.interp(np.linspace(0, 1, out_len), np.linspace(0, 1, seg_n), sig[i:i + seg_n])
        sd = seg1k.std()
        if sd > std_thr:
            segs.append(((seg1k - seg1k.mean()) / sd).astype(np.float32))
            labels.append(lab)
    return (np.array(segs), np.array(labels, dtype=np.int64))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mitdb', default='E:\\code\\biology1.03\\data\\mitdb')
    ap.add_argument('--out', default='E:\\code\\biology1.03\\data\\processed_real')
    ap.add_argument('--n-clients', type=int, default=6)
    ap.add_argument('--per-client', type=int, default=125, help='')
    ap.add_argument('--balance', action='store_true', help='')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--win', type=int, default=1000)
    ap.add_argument('--scheme', choices=['aami', 'record'], default='aami', help='')
    a = ap.parse_args()
    from sklearn.model_selection import train_test_split
    rng = np.random.RandomState(a.seed)
    mitdb = Path(a.mitdb)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = sorted([p.stem for p in mitdb.glob('*.hea')])
    if a.scheme == 'record':
        recs = [r for r in recs if r in RECORD_TO_CLASS]
    per = max(1, len(recs) // a.n_clients)
    clients = {i: recs[i * per:(i + 1) * per if i < a.n_clients - 1 else len(recs)] for i in range(a.n_clients)}
    for cid, rec_list in clients.items():
        all_x, all_y = ([], [])
        for r in rec_list:
            try:
                if a.scheme == 'record':
                    x, y = extract_record_level(mitdb / r)
                else:
                    x, y = extract_record(mitdb / r, win=a.win)
                all_x.append(x)
                all_y.append(y)
            except Exception as e:
                print(f'{r}: {e}')
        if not all_x:
            print(f'  Client {cid}')
            continue
        X = np.concatenate(all_x, 0)
        y = np.concatenate(all_y, 0)
        if a.balance:
            per_cls = max(1, a.per_client // 5)
            idx = []
            for c in range(5):
                ci = np.where(y == c)[0]
                if len(ci) == 0:
                    continue
                take = min(per_cls, len(ci))
                idx.extend(rng.choice(ci, take, replace=False).tolist())
            idx = np.array(idx, dtype=np.int64)
            n = len(idx)
        else:
            n = min(a.per_client, len(X))
            idx = rng.choice(len(X), n, replace=False) if len(X) > n else np.arange(len(X))
        X, y = (X[idx], y[idx])
        X = X[:, np.newaxis, :]
        y = y.astype(np.int64)
        try:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=a.seed, stratify=y)
        except ValueError:
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=a.seed)
        np.save(out / f'ecg_client{cid}_train_X.npy', Xtr)
        np.save(out / f'ecg_client{cid}_train_y.npy', ytr)
        np.save(out / f'ecg_client{cid}_test_X.npy', Xte)
        np.save(out / f'ecg_client{cid}_test_y.npy', yte)
        print(f'  Client {cid}: {len(rec_list)}{n}{len(Xtr)} / test {len(Xte)}{np.bincount(y, minlength=5)}')
if __name__ == '__main__':
    main()