# Reference implementation for the LBRD-PPFL manuscript.

import argparse
import importlib.util
import sys
import time
from pathlib import Path
import h5py
import numpy as np
import torch
PROJ = Path('E:\\code\\biology1.03')
sys.path.insert(0, str(PROJ))
_spec = importlib.util.spec_from_file_location('u10', str(PROJ / '10_rerun_unified.py'))
u10 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(u10)
from models.llm_guide import LLMGuide, DeepSeekConfig
from models.lbrd_generator import LBRDGenerator, LBRDConfig
PREFIX = {'ECG': 'ecg', 'PPG': 'ppg', 'EEG': 'eeg'}

def build_model(device, num_classes=5):
    return u10.SimpleCNN(1, 1000, [32, 64, 128], num_classes, use_batchnorm=True, dropout_rate=0.2).to(device)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=list(PREFIX))
    ap.add_argument('--real_dir', default=str(PROJ / 'data' / 'processed_real'))
    ap.add_argument('--per-client', type=int, default=125, help='')
    ap.add_argument('--n-clients', type=int, default=6)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--out', default=str(PROJ / 'lbrd_cache' / 'real_robust'))
    ap.add_argument('--offline', action='store_true', help='')
    ap.add_argument('--no-semantic', action='store_true', help='')
    ap.add_argument('--no-visual', action='store_true', help='')
    ap.add_argument('--no-gradient', action='store_true', help='')
    ap.add_argument('--tag', default=None, help='')
    a = ap.parse_args()
    if a.offline:
        import os
        os.environ['DEEPSEEK_API_KEY'] = ''
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    prefix = PREFIX[a.dataset]
    real_dir = Path(a.real_dir)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{prefix}_robust{('_' + a.tag if a.tag else '')}.hdf5"
    if out.exists():
        out.unlink()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = LBRDConfig()
    cfg.postprocess_match_stats = False
    if a.no_semantic:
        cfg.use_semantic = False
    if a.no_visual:
        cfg.use_visual = False
    if a.no_gradient:
        cfg.lambda1 = 0.0
    print(f'[ablation] use_grad={cfg.lambda1 > 0} use_struct={cfg.use_structure} use_semantic={cfg.use_semantic} use_visual={cfg.use_visual}')
    guide = LLMGuide(bio_type=a.dataset, config=DeepSeekConfig())
    gen = LBRDGenerator(llm_guide=guide, bio_type=a.dataset, device=dev, config=cfg)
    model = build_model(dev)
    print(f"=== dataset={a.dataset} per_client={a.per_client} device={dev} api_key={('' if guide.config.api_key else '')} ===")
    t_all = time.time()
    for c in range(a.n_clients):
        tr_x = np.load(real_dir / f'{prefix}_client{c}_train_X.npy').astype(np.float32)
        tr_y = np.load(real_dir / f'{prefix}_client{c}_train_y.npy').astype(np.int64)
        n = min(a.per_client, len(tr_x))
        idx = np.random.RandomState(a.seed + c).choice(len(tr_x), n, replace=False)
        x = torch.FloatTensor(tr_x[idx]).to(dev)
        y = torch.LongTensor(tr_y[idx]).to(dev)
        t0 = time.time()
        rob = gen.generate_robust_data(x, model, labels=y)
        dt = time.time() - t0
        rob_np = rob.detach().cpu().numpy().astype(np.float32)
        rel = float((rob - x).norm() / (x.norm() + 1e-12))
        comp = dict(getattr(gen, 'last_loss_components', {}) or {})
        print(f'client_{c}: {n}{dt:.0f}{rel:.4f}{gen.last_postprocess_fallback}')
        if comp:
            print('' + '  '.join((f'{k}={v:.4f}' for k, v in comp.items())))
        with h5py.File(out, 'a') as w:
            g = w.create_group(f'client_{c}')
            g.create_dataset('raw_data', data=tr_x[idx], compression='gzip')
            g.create_dataset('raw_labels', data=tr_y[idx], compression='gzip')
            g.create_dataset('robust_data', data=rob_np, compression='gzip')
            g.create_dataset('robust_labels', data=tr_y[idx], compression='gzip')
            if comp:
                import json as _json
                g.attrs['loss_components'] = _json.dumps(comp, ensure_ascii=False)
            import json as _json2
            g.attrs['ablation'] = _json2.dumps({'use_grad': bool(cfg.lambda1 > 0), 'use_structure': bool(cfg.use_structure), 'use_semantic': bool(cfg.use_semantic), 'use_visual': bool(cfg.use_visual)})
    print(f'=== done in {(time.time() - t_all) / 60:.1f} min -> {out} ===')
if __name__ == '__main__':
    main()