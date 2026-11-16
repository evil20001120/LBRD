# Reference implementation for the LBRD-PPFL manuscript.

from __future__ import annotations
import hashlib
import math
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
__all__ = ['TextEncoderConfig', 'SemanticTextEncoder', 'SemanticProjector', 'StatisticalFeatureEncoder', 'create_semantic_text_encoder', 'text_embedding_regularizer', 'variance_regularizer', 'init_from_corpus_stats', 'info_nce_loss', 'queue_push']
_TOKEN_RE = re.compile('[A-Za-z]+|\\d+|[^\\sA-Za-z\\d]')

@dataclass
class TextEncoderConfig:
    vocab_size: int = 8192
    embed_dim: int = 256
    out_dim: int = 768
    max_len: int = 128
    dropout: float = 0.1

class SemanticTextEncoder(nn.Module):

    def __init__(self, config: Optional[TextEncoderConfig]=None):
        super().__init__()
        self.config = config or TextEncoderConfig()
        c = self.config
        self.embed = nn.Embedding(c.vocab_size, c.embed_dim, padding_idx=0)
        self.dropout = nn.Dropout(c.dropout)
        self.proj = nn.Sequential(nn.Linear(c.embed_dim, c.embed_dim), nn.GELU(), nn.Linear(c.embed_dim, c.out_dim), nn.LayerNorm(c.out_dim))
        nn.init.normal_(self.embed.weight, mean=0.0, std=c.embed_dim ** (-0.5))
        with torch.no_grad():
            self.embed.weight[0].zero_()
        for m in self.proj:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def _token_ids(self, text: str) -> List[int]:
        c = self.config
        toks = _TOKEN_RE.findall(text or '')[:c.max_len]
        if not toks:
            return [1]
        ids = []
        for t in toks:
            h = hashlib.blake2b(t.encode('utf-8'), digest_size=8).digest()
            ids.append(int.from_bytes(h, 'big') % (c.vocab_size - 1) + 1)
        return ids

    def tokenize(self, texts: Union[str, Sequence[str]]) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        seqs = [self._token_ids(t) for t in texts]
        L = max((len(s) for s in seqs))
        out = torch.zeros(len(seqs), L, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        return out

    def forward(self, texts: Union[str, Sequence[str]], device: Optional[torch.device]=None) -> torch.Tensor:
        ids = self.tokenize(texts)
        if device is not None:
            ids = ids.to(device)
        elif self.embed.weight.is_cuda:
            ids = ids.to(self.embed.weight.device)
        pad_mask = (ids != 0).float().unsqueeze(-1)
        emb = self.dropout(self.embed(ids))
        summed = (emb * pad_mask).sum(dim=1)
        counts = pad_mask.sum(dim=1).clamp(min=1.0)
        pooled = summed / counts
        return self.proj(pooled)

class SemanticProjector(nn.Module):

    def __init__(self, in_dim: int, out_dim: int=768, hidden: Optional[int]=None):
        super().__init__()
        hidden = hidden or out_dim
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim), nn.LayerNorm(out_dim))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)

def create_semantic_text_encoder(config: Optional[TextEncoderConfig]=None) -> SemanticTextEncoder:
    return SemanticTextEncoder(config)

def text_embedding_regularizer(emb: torch.Tensor, eps: float=0.0001, cos_threshold: float=0.5) -> torch.Tensor:
    if emb.dim() == 1:
        emb = emb.unsqueeze(0)
    if emb.size(0) < 2:
        return torch.zeros((), device=emb.device, requires_grad=True)
    x = emb - emb.mean(dim=0, keepdim=True)
    std = torch.sqrt(x.var(dim=0, unbiased=False) + eps)
    scale = emb.detach().std(dim=1, unbiased=False).mean() + eps
    var_term = F.relu(1.0 - std / scale).mean()
    z = F.normalize(emb, p=2, dim=-1)
    gram = z @ z.t()
    n = z.size(0)
    off_diag = gram - torch.diag_embed(torch.diagonal(gram))
    cos_term = F.relu(off_diag - cos_threshold).sum() / max(n * (n - 1), 1)
    return var_term + cos_term

def variance_regularizer(emb: torch.Tensor, eps: float=0.0001) -> torch.Tensor:
    if emb.dim() == 1:
        emb = emb.unsqueeze(0)
    if emb.size(0) < 2:
        return torch.zeros((), device=emb.device, requires_grad=True)
    x = emb - emb.mean(dim=0, keepdim=True)
    std = torch.sqrt(x.var(dim=0, unbiased=False) + eps)
    scale = emb.detach().std(dim=1, unbiased=False).mean() + eps
    return F.relu(1.0 - std / scale).mean()

class StatisticalFeatureEncoder(nn.Module):
    _MU = (-2.419384, 4.814325, 2.30259, 1.873386, 0.014, 0.03003, 0.105036, 0.753756)
    _SIGMA = (1.307047, 1.259011, 0.573091, 1.675328, 0.008895, 0.013542, 0.028104, 0.049205)

    def __init__(self, out_dim: int=768, seed: int=0):
        super().__init__()
        self.out_dim = out_dim
        g = torch.Generator().manual_seed(seed)
        self.register_buffer('W', torch.randn(out_dim, 8, generator=g))
        self.register_buffer('b', torch.rand(out_dim, generator=g) * (2.0 * math.pi))
        self.register_buffer('mu', torch.tensor(self._MU, dtype=torch.float32))
        self.register_buffer('sigma', torch.tensor(self._SIGMA, dtype=torch.float32))

    @staticmethod
    def _stats(x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.size(0), -1)
        L = x.size(1)
        mn = x.min(dim=1).values
        mx = x.max(dim=1).values
        log_kurt = torch.log((x ** 4).mean(dim=1).clamp(min=1e-12))
        skew = (x ** 3).mean(dim=1)
        X = torch.fft.rfft(x, dim=1)
        mag = X.abs()
        freqs = torch.fft.rfftfreq(L, device=x.device)
        dom = freqs[mag[:, 1:].argmax(dim=1) + 1]
        sgn = torch.sign(x)
        zcr = (sgn[:, 1:] * sgn[:, :-1] < 0).float().mean(dim=1)
        m = mag[:, 1:]
        p = m / (m.sum(dim=1, keepdim=True) + 1e-12)
        sc = 2.0 * (p * freqs[1:]).sum(dim=1)
        se = -(p * (p + 1e-12).log()).sum(dim=1) / math.log(m.size(1))
        return torch.stack([mn, mx, log_kurt, skew, dom * 2.0, zcr, sc, se], dim=1)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = (self._stats(x) - self.mu) / self.sigma
        f = f.clamp(-5.0, 5.0)
        z = torch.cos(f @ self.W.t() + self.b) * (2.0 / self.out_dim) ** 0.5
        return F.normalize(z, p=2, dim=-1)

def _char_phi(tok: str) -> np.ndarray:
    phi = np.zeros(38, dtype=np.float64)
    n = max(len(tok), 1)
    for ch in tok.lower():
        if 'a' <= ch <= 'z':
            phi[ord(ch) - 97] += 1.0
    phi[:26] /= n
    phi[26] = min(len(tok), 16) / 16.0
    if tok.isdigit():
        for ch in tok:
            phi[27 + (ord(ch) - 48)] += 1.0
        phi[27:37] /= n
        phi[37] = math.log1p(int(tok)) / math.log1p(1000000.0)
    raw = hashlib.blake2b(tok.encode('utf-8'), digest_size=16).digest()
    rid = np.random.default_rng(int.from_bytes(raw, 'big')).standard_normal(16) / 4.0
    return np.concatenate([phi, rid])

def init_from_corpus_stats(encoder: SemanticTextEncoder, texts: Sequence[str], seed: int=0) -> Dict[str, float]:
    C = encoder.config
    D = C.embed_dim
    PHI_D = 54
    g = torch.Generator().manual_seed(seed)
    P = torch.randn(D, PHI_D, generator=g) / PHI_D ** 0.5
    acc_vec = np.zeros((C.vocab_size, PHI_D), dtype=np.float64)
    acc_cnt = np.zeros(C.vocab_size, dtype=np.float64)
    for text in texts:
        for tok in _TOKEN_RE.findall(text or '')[:C.max_len]:
            bid = int.from_bytes(hashlib.blake2b(tok.encode('utf-8'), digest_size=8).digest(), 'big') % (C.vocab_size - 1) + 1
            acc_vec[bid] += _char_phi(tok)
            acc_cnt[bid] += 1.0
    hit = acc_cnt > 0
    phi_avg = np.zeros((C.vocab_size, PHI_D), dtype=np.float64)
    phi_avg[hit] = acc_vec[hit] / acc_cnt[hit, None]
    phi_all = acc_vec.sum(axis=0) / max(acc_cnt.sum(), 1.0)
    phi_avg[1] = phi_all
    emb = phi_avg @ P.numpy().T
    base_std = float(D) ** (-0.5)
    noise = torch.randn(C.vocab_size, D, generator=g) * base_std
    hm = torch.from_numpy(hit)
    emb_t = torch.from_numpy(emb).float()
    emb_t[~hm] = noise[~hm]
    target_std = base_std
    cur_std = float(emb_t[hm].std()) if hit.any() else 1.0
    if cur_std > 0:
        emb_t[hm] *= target_std / (cur_std + 1e-09)
    with torch.no_grad():
        w = encoder.embed.weight
        w.copy_(emb_t.to(w.dtype).to(w.device))
        w[0].zero_()
    return {'corpus_texts': len(texts), 'corpus_tokens': float(acc_cnt.sum()), 'hit_buckets': float(hit.sum()), 'vocab_size': float(C.vocab_size), 'target_std': target_std, 'applied_std': float(emb_t[hm].std()) if hit.any() else 0.0}

def info_nce_loss(anchor: torch.Tensor, positive: torch.Tensor, negatives: Optional[torch.Tensor]=None, tau: float=0.07) -> torch.Tensor:
    a = F.normalize(anchor, p=2, dim=-1)
    p = F.normalize(positive, p=2, dim=-1)
    logits = a @ p.t() / tau
    if negatives is not None and negatives.numel() > 0 and (negatives.size(0) > 0):
        q = F.normalize(negatives.to(a.device), p=2, dim=-1)
        logits = torch.cat([logits, a @ q.t() / tau], dim=1)
    labels = torch.arange(a.size(0), device=a.device)
    return F.cross_entropy(logits, labels)

def queue_push(queue: Optional[torch.Tensor], new: torch.Tensor, cap: int) -> torch.Tensor:
    new = new.detach()
    if queue is None or queue.numel() == 0:
        q = new
    else:
        q = torch.cat([queue.to(new.device), new], dim=0)
    if q.size(0) > cap:
        q = q[-cap:]
    return q.detach()