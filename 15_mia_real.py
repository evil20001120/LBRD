# Reference implementation for the LBRD-PPFL manuscript.

import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
PROJ = Path(__file__).resolve().parent
if not (PROJ / '10_rerun_unified.py').exists():
    PROJ = Path('E:\\code\\biology1.03')
sys.path.insert(0, str(PROJ))
_spec = importlib.util.spec_from_file_location('u10', str(PROJ / '10_rerun_unified.py'))
u10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(u10)
_spec12 = importlib.util.spec_from_file_location('r12', str(PROJ / '12_rerun_real.py'))
r12 = importlib.util.module_from_spec(_spec12)
import types
r12.__dict__['__name__'] = 'r12'
_spec12.loader.exec_module(r12)
PREFIX = {'ECG': 'ecg', 'PPG': 'ppg', 'EEG': 'eeg'}
METHODS = ['fedavg', 'dpfedavg', 'fedprox', 'dpfedcgan', 'lbrd', 'lbrd_dp']

def per_sample_loss(model, x, y, bs=128):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), bs):
            xb, yb = (x[i:i + bs], y[i:i + bs])
            out.append(F.cross_entropy(model(xb), yb, reduction='none').detach().cpu().numpy())
    return np.concatenate(out) if out else np.array([])

def roc_auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    uniq, inv, cnt = np.unique(scores, return_inverse=True, return_counts=True)
    if (cnt > 1).any():
        s = np.zeros(len(uniq))
        np.add.at(s, inv, ranks)
        ranks = (s / cnt)[inv]
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))

def best_advantage(scores, labels):
    best = -1.0
    for t in np.unique(scores):
        pred = scores >= t
        tpr = (pred & (labels == 1)).sum() / max((labels == 1).sum(), 1)
        fpr = (pred & (labels == 0)).sum() / max((labels == 0).sum(), 1)
        best = max(best, abs(tpr - fpr))
    return float(best)

def tpr_at_fpr(scores, labels, target=0.1):
    """TPR at the maximum FPR not exceeding `target`.

    Returns the *highest* TPR among all thresholds whose FPR <= target.
    NOTE: an earlier version returned at the first threshold satisfying
    FPR <= target, which is the highest score point (FPR = 0) and therefore
    systematically returned (near-)zero TPR.  Scan every admissible threshold
    and keep the best TPR instead.
    """
    n_pos = max(int((labels == 1).sum()), 1)
    n_neg = max(int((labels == 0).sum()), 1)
    best = 0.0
    for t in np.unique(scores)[::-1]:
        pred = scores >= t
        fpr = (pred & (labels == 0)).sum() / n_neg
        if fpr > target:
            continue
        tpr = (pred & (labels == 1)).sum() / n_pos
        if tpr > best:
            best = tpr
    return float(best)

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
    ap.add_argument('--sigma', type=float, default=None)
    ap.add_argument('--clip_norm', type=float, default=1.0)
    ap.add_argument('--real_dir', default=None)
    ap.add_argument('--robust_cache', default=None)
    ap.add_argument('--lbrd_mode', choices=['augment', 'replace'], default='augment', help='')
    ap.add_argument('--member_source', choices=['raw', 'robust', 'replaced'], default='raw', help='')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--out', default=str(Path('E:\\wb_tmp\\lbrd\\mia_real')))
    a = ap.parse_args()
    if a.real_dir:
        r12._REAL_DIR = Path(a.real_dir)
    prefix = PREFIX[a.dataset]
    u10.DATASETS[a.dataset] = dict(cache='<processed_real>', input_channels=1, seq_length=1000, num_classes=5, hidden_dims=[32, 64, 128])
    cfg = u10.Cfg(a.dataset, num_rounds=a.rounds, local_epochs=a.local_epochs, batch_size=a.batch_size, lr=a.lr, seed=a.seed, epsilon=a.epsilon, delta=a.delta, clip_norm=a.clip_norm)
    u10.load_cache = lambda path, c, ur, pr, rc=None: r12.load_real(prefix, c, ur, pr, rc, r12._LBRD_MODE)
    r12._LBRD_MODE = a.lbrd_mode
    tag = a.tag or f'MIA_{a.dataset}_{a.method}_s{a.seed}'
    res, model = u10.run(cfg, a.method, a.robust_cache, a.sigma, tag, return_model=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_robust = a.method in ('lbrd', 'lbrd_dp')
    cache = '<processed_real>'
    clients = u10.load_cache(cache, cfg, use_robust, 'noleak', a.robust_cache)
    clients = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in c.items()} for c in clients]
    mem_s, non_s, mem_l, non_l = ([], [], [], [])
    rb = {}
    if a.member_source == 'replaced':
        if not a.robust_cache:
            raise SystemExit('')
        rb = r12._load_robust_raw(a.robust_cache)
    for ci, cl in enumerate(clients):
        if a.member_source == 'replaced':
            if ci not in rb:
                continue
            mx = torch.FloatTensor(rb[ci][0]).to(device)
            if mx.dim() == 2:
                mx = mx.unsqueeze(1)
            my = torch.LongTensor(rb[ci][1]).to(device)
        else:
            n_raw = int(cl['train_size'] - cl.get('n_robust_kept', 0))
            if a.member_source == 'robust':
                mx, my = (cl['train_data'][n_raw:], cl['train_labels'][n_raw:])
                if len(mx) == 0:
                    continue
            else:
                mx, my = (cl['train_data'][:n_raw], cl['train_labels'][:n_raw])
        l_tr = per_sample_loss(model, mx, my)
        l_te = per_sample_loss(model, cl['test_data'], cl['test_labels'])
        mem_s.append(-l_tr)
        non_s.append(-l_te)
        mem_l.append(np.ones(len(l_tr)))
        non_l.append(np.zeros(len(l_te)))
    scores = np.concatenate(mem_s + non_s)
    labels = np.concatenate(mem_l + non_l)
    auc = roc_auc(scores, labels)
    adv = best_advantage(scores, labels)
    tpr = tpr_at_fpr(scores, labels, 0.1)
    tpr1 = tpr_at_fpr(scores, labels, 0.01)
    tpr01 = tpr_at_fpr(scores, labels, 0.001)
    tr_loss = float(np.mean(-np.concatenate(mem_s)))
    te_loss = float(np.mean(-np.concatenate(non_s)))
    try:
        cm = np.zeros((int(u10.DATASETS[a.dataset]['num_classes']), int(u10.DATASETS[a.dataset]['num_classes'])), dtype=np.int64)
        for cl in clients:
            xt = cl['test_data']
            yt = cl['test_labels'].detach().cpu().numpy()
            model.eval()
            with torch.no_grad():
                preds = []
                for i in range(0, len(xt), 256):
                    preds.append(model(xt[i:i + 256]).argmax(dim=1).detach().cpu().numpy())
            yp = np.concatenate(preds) if preds else np.array([], dtype=np.int64)
            for t_, p_ in zip(yt, yp):
                cm[int(t_), int(p_)] += 1
        cm_list = cm.tolist()
    except Exception as _e:
        print(f'[warn] confusion matrix failed: {_e}')
        cm_list = None
    out = dict(res)
    out.update({'mia_auc': auc, 'mia_advantage': adv, 'tpr_at_fpr10': tpr, 'tpr_at_fpr1': tpr1, 'tpr_at_fpr01': tpr01, 'confusion_matrix': cm_list, 'member_source': a.member_source, 'lbrd_mode': a.lbrd_mode, 'mi_member_n': int(labels.sum()), 'mi_nonmember_n': int(len(labels) - labels.sum()), 'member_loss': tr_loss, 'nonmember_loss': te_loss, 'train_test_gap': tr_loss - te_loss})
    print(f'[{tag}] MIA AUC={auc:.4f}  Advantage={adv:.4f}  TPR@FPR10%={tpr:.4f}  TPR@FPR1%={tpr1:.4f}  TPR@FPR0.1%={tpr01:.4f}  member={a.member_source} n={int(labels.sum())}  member_loss={tr_loss:.4f} nonmember_loss={te_loss:.4f} gap={tr_loss - te_loss:+.4f}')
    if cm_list is not None:
        _cm = np.asarray(cm_list)
        print(f'[{tag}] confusion matrix (rows=true, cols=pred):\n{_cm}')
    d = Path(a.out)
    d.mkdir(parents=True, exist_ok=True)
    (d / f'{tag}.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f"[{tag}] saved -> {d / (tag + '.json')}")
if __name__ == '__main__':
    main()