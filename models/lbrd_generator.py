# Reference implementation for the LBRD-PPFL manuscript.

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import pickle
import warnings
from dataclasses import dataclass
import logging
try:
    from .semantic_encoder import SemanticTextEncoder, SemanticProjector, TextEncoderConfig, text_embedding_regularizer, StatisticalFeatureEncoder, init_from_corpus_stats, info_nce_loss, queue_push, variance_regularizer
except ImportError:
    from semantic_encoder import SemanticTextEncoder, SemanticProjector, TextEncoderConfig, text_embedding_regularizer, StatisticalFeatureEncoder, init_from_corpus_stats, info_nce_loss, queue_push, variance_regularizer
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LBRDConfig:
    max_iterations: int = 50
    learning_rate: float = 0.003
    epsilon: float = 0.1
    alpha: float = 0.3
    beta: float = 1.0
    gamma: float = 0.6
    lambda1: float = 1.0
    lambda2: float = 0.8
    lambda3: float = 0.5
    early_stop_patience: int = 10
    gradient_clip: float = 1.0
    semantic_weight: float = 0.5
    structure_weight: float = 0.3
    noise_weight: float = 0.2
    use_semantic: bool = True
    use_structure: bool = True
    use_visual: bool = True
    cache_enabled: bool = True
    cache_size: int = 100
    constraint_epsilon: float = 0.1
    perturbation_strength: float = 0.3
    adaptive_lr: bool = True
    lr_decay_factor: float = 0.95
    min_lr: float = 0.0001
    semantic_embed_dim: int = 768
    semantic_margin: float = 0.2
    gradient_second_order: bool = True
    visual_detach_target: bool = True
    use_learnable_text_encoder: bool = True
    visual_margin: float = 1.0
    visual_margin_weight: float = 1.0
    text_regularizer_weight: float = 0.1
    text_encoder_trainable: bool = False
    gradient_relative_weight: float = 1.0
    visual_margin_mode: str = 'ratio_of_logc'
    visual_margin_ratio: float = 0.1
    postprocess_match_stats: bool = True
    semantic_source: str = 'llm_text'
    text_encoder_init: str = 'random'
    stats_prior_texts: Optional[List[str]] = None
    use_infonce: bool = False
    infonce_weight: float = 1.0
    infonce_tau: float = 0.07
    infonce_queue_size: int = 256
    infonce_var_weight: float = 1.0
    text_full_batch: bool = True
    structure_block_weights: Optional[List[float]] = None
ECG_BLOCK_SPEC = [('P', [(420, 470)]), ('PR', [(470, 486)]), ('QRS', [(486, 522)]), ('ST', [(522, 560)]), ('T', [(560, 650)]), ('QT', [(486, 650)]), ('HRV', [(0, 1000)]), ('baseline', [(0, 420), (650, 1000)])]

class LBRDGenerator(nn.Module):
    _feature_dim: int = 64

    def __init__(self, llm_guide, bio_type: str='ECG', device: str='cuda', config: LBRDConfig=None):
        super().__init__()
        self.llm_guide = llm_guide
        self.bio_type = bio_type.upper()
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.config = config or LBRDConfig()
        self.generator = self._build_generator().to(self.device)
        self.text_encoder = SemanticTextEncoder(TextEncoderConfig(out_dim=self.config.semantic_embed_dim)).to(self.device)
        self.w_proj = SemanticProjector(in_dim=self._feature_dim, out_dim=self.config.semantic_embed_dim).to(self.device)
        if self.config.semantic_source == 'stat_feature':
            self.stat_encoder = StatisticalFeatureEncoder(out_dim=self.config.semantic_embed_dim).to(self.device)
        else:
            self.stat_encoder = None
        self.last_prior_init_info: Dict[str, Any] = {}
        if self.config.text_encoder_init == 'stats_prior' and self.config.stats_prior_texts:
            self.last_prior_init_info = init_from_corpus_stats(self.text_encoder, list(self.config.stats_prior_texts))
            logger.info('', self.last_prior_init_info.get('hit_buckets'), self.last_prior_init_info.get('vocab_size'), self.last_prior_init_info.get('corpus_tokens'))
        self._infonce_queue: Optional[torch.Tensor] = None
        if self.config.use_infonce:
            self.text_encoder.train()
        try:
            if self.config.use_learnable_text_encoder:
                self.llm_guide.set_text_encoder(self.text_encoder)
        except AttributeError:
            pass
        if not self.config.text_encoder_trainable:
            self.text_encoder.requires_grad_(False)
            self.text_encoder.eval()
        _params = list(self.generator.parameters()) + list(self.w_proj.parameters())
        if self.config.text_encoder_trainable:
            _params += list(self.text_encoder.parameters())
        self.optimizer = torch.optim.Adam(_params, lr=self.config.learning_rate)
        self.last_loss_gradnorms: Dict[str, float] = {}
        self.last_loss_components: Dict[str, float] = {}
        self.last_visual_margin: float = 0.0
        self._num_classes_hint: Optional[int] = None
        self.last_perturbation_before_post: float = 0.0
        self.last_perturbation_after_post: float = 0.0
        self.last_postprocess_fallback: bool = False
        self.cache = {}
        self.cache_size = self.config.cache_size
        self.generation_history = {'losses': [], 'semantic_scores': [], 'structure_scores': [], 'visual_scores': [], 'iterations': []}
        logger.info(f'LBRD Generator initialized for {self.bio_type} on {self.device}')

    def _build_generator(self) -> nn.Module:

        class BioGenerator(nn.Module):

            def __init__(self, input_channels=1, output_channels=1):
                super().__init__()
                self.encoder = nn.Sequential(nn.Conv1d(input_channels, 16, kernel_size=7, padding=3), nn.ReLU(), nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(), nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU())
                self.decoder = nn.Sequential(nn.ConvTranspose1d(64, 32, kernel_size=3, padding=1), nn.ReLU(), nn.ConvTranspose1d(32, 16, kernel_size=5, padding=2), nn.ReLU(), nn.ConvTranspose1d(16, output_channels, kernel_size=7, padding=3), nn.Tanh())

            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                if decoded.size(-1) != x.size(-1):
                    decoded = F.interpolate(decoded, size=x.size(-1), mode='linear', align_corners=False)
                return decoded

            def encode(self, x):
                return self.encoder(x).mean(dim=-1)
        return BioGenerator()

    def generate_robust_data(self, original_data: torch.Tensor, model: nn.Module, target_class: int=None, current_round: int=0, labels: Optional[torch.Tensor]=None) -> torch.Tensor:
        self.generator.train()
        try:
            semantic_guidance: Optional[torch.Tensor] = None
            if self.config.semantic_source != 'stat_feature':
                with torch.no_grad():
                    semantic_guidance = self.llm_guide.generate_semantic_guidance(original_data.detach(), target_class, guidance_strength=0.5).to(self.device)
            semantic_texts: Optional[List[str]] = None
            if self.config.semantic_source != 'stat_feature' and self.config.use_semantic and (self.config.semantic_embed_dim > 0):
                try:
                    getter = getattr(self.llm_guide, 'get_semantic_texts', None)
                    if callable(getter):
                        max_batch = int(original_data.size(0)) if self.config.text_full_batch else 5
                        semantic_texts = getter(original_data.detach(), max_batch=max_batch)
                except Exception as e:
                    semantic_texts = None
            best_robust = None
            best_loss = float('inf')
            patience_counter = 0
            for iteration in range(self.config.max_iterations):
                self.optimizer.zero_grad()
                with torch.no_grad():
                    noise = torch.randn_like(original_data) * 0.1
                    input_with_noise = original_data + noise
                candidate = self.generator(input_with_noise)
                total_loss = self._compute_generation_loss(original_data.detach(), candidate, model, semantic_guidance.detach() if semantic_guidance is not None else None, iteration, semantic_texts=semantic_texts, target_labels=labels)
                if not torch.isfinite(total_loss):
                    logger.warning(f'Iteration {iteration}: Loss is NaN/Inf, skipping')
                    continue
                self.optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                try:
                    model.zero_grad(set_to_none=True)
                except Exception:
                    pass
                torch.nn.utils.clip_grad_norm_([p for grp in self.optimizer.param_groups for p in grp['params']], self.config.gradient_clip)
                self.optimizer.step()
                if iteration == 0 or total_loss.item() < best_loss:
                    best_loss = total_loss.item()
                    best_robust = candidate.detach().clone()
                    patience_counter = 0
                else:
                    patience_counter += 1
                self.generation_history['losses'].append(total_loss.item())
                self.generation_history['iterations'].append(iteration)
                if patience_counter >= self.config.early_stop_patience:
                    break
                if self.config.adaptive_lr and iteration % 10 == 0:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = max(param_group['lr'] * self.config.lr_decay_factor, self.config.min_lr)
            if best_robust is None:
                best_robust = candidate.detach()
            robust_data = self._postprocess_robust_data(original_data, best_robust)
            logger.debug(f'Generated robust data: {robust_data.shape}')
            return robust_data
        except Exception as e:
            logger.error(f'Generation failed: {e}')
            import traceback
            traceback.print_exc()
            return original_data.clone()

    def _compute_generation_loss(self, original: torch.Tensor, candidate: torch.Tensor, model: nn.Module, semantic_guidance: Optional[torch.Tensor]=None, iteration: int=0, semantic_texts: Optional[List[str]]=None, target_labels: Optional[torch.Tensor]=None) -> torch.Tensor:
        loss_components = {}
        if self.config.lambda1 > 0:
            try:
                grad_loss = self._gradient_utility_loss(original, candidate, model, target_labels=target_labels)
                if torch.isfinite(grad_loss):
                    loss_components['gradient'] = self.config.lambda1 * grad_loss
                else:
                    loss_components['gradient'] = torch.zeros((), device=self.device)
            except Exception as e:
                logger.warning(f'Gradient loss failed: {e}')
                loss_components['gradient'] = torch.zeros((), device=self.device)
        if self.config.use_structure and self.config.lambda2 > 0:
            try:
                struct_loss = self._structure_preservation_loss(original, candidate)
                if torch.isfinite(struct_loss):
                    loss_components['structure'] = self.config.lambda2 * struct_loss
                else:
                    loss_components['structure'] = torch.tensor(0.0, device=self.device)
            except Exception as e:
                logger.warning(f'Structure loss failed: {e}')
                loss_components['structure'] = torch.tensor(0.0, device=self.device)
        if self.config.use_semantic and self.config.gamma > 0:
            try:
                sem_loss = self._semantic_privacy_loss(original, candidate, semantic_guidance, semantic_texts=semantic_texts)
                if torch.isfinite(sem_loss):
                    loss_components['semantic'] = self.config.gamma * sem_loss
                else:
                    loss_components['semantic'] = torch.zeros((), device=self.device)
            except Exception as e:
                logger.warning(f'Semantic loss failed: {e}')
                loss_components['semantic'] = torch.zeros((), device=self.device)
        if self.config.use_visual and self.config.beta > 0:
            try:
                vis_loss = self._visual_privacy_loss(original, candidate, model)
                if torch.isfinite(vis_loss):
                    margin = self._resolve_visual_margin(model)
                    push = F.relu(margin - vis_loss)
                    loss_components['visual'] = self.config.beta * self.config.visual_margin_weight * push
                    self.last_visual_margin = float(margin)
                else:
                    loss_components['visual'] = torch.zeros((), device=self.device)
            except Exception as e:
                logger.warning(f'Visual loss failed: {e}')
                loss_components['visual'] = torch.zeros((), device=self.device)
        if self.config.text_encoder_trainable and self.config.use_semantic and semantic_texts:
            if self.config.use_infonce and self.config.infonce_weight > 0:
                try:
                    emb_t = self.text_encoder(semantic_texts, device=self.device)
                    with torch.no_grad():
                        pos = F.normalize(self.w_proj(self.generator.encode(candidate)), p=2, dim=-1)
                    ice = info_nce_loss(emb_t, pos, self._infonce_queue, tau=float(self.config.infonce_tau))
                    var_reg = variance_regularizer(emb_t)
                    if torch.isfinite(ice) and torch.isfinite(var_reg):
                        loss_components['infonce'] = self.config.infonce_weight * ice + self.config.infonce_var_weight * var_reg
                        self._infonce_queue = queue_push(self._infonce_queue, pos.detach(), int(self.config.infonce_queue_size))
                except Exception as e:
                    logger.warning(f'InfoNCE loss failed: {e}')
            elif self.config.text_regularizer_weight > 0:
                try:
                    emb = self.text_encoder(semantic_texts, device=self.device)
                    reg = text_embedding_regularizer(emb)
                    loss_components['text_reg'] = self.config.text_regularizer_weight * reg
                except Exception as e:
                    logger.warning(f'Text regularizer failed: {e}')
        try:
            perturb_loss = self._perturbation_constraint(original, candidate)
            if torch.isfinite(perturb_loss):
                loss_components['perturbation'] = self.config.epsilon * perturb_loss
            else:
                loss_components['perturbation'] = torch.tensor(0.0, device=self.device)
        except Exception as e:
            logger.warning(f'Perturbation loss failed: {e}')
            loss_components['perturbation'] = torch.tensor(0.0, device=self.device)
        total_loss = sum(loss_components.values())
        if not torch.isfinite(total_loss):
            logger.warning(f'Total loss is not finite: {loss_components}')
            return torch.tensor(1.0, device=self.device, requires_grad=True)
        try:
            self.last_loss_components = {k: float(v.detach()) for k, v in loss_components.items()}
        except Exception:
            pass
        return total_loss

    def _gradient_utility_loss(self, original: torch.Tensor, candidate: torch.Tensor, model: nn.Module, target_labels: Optional[torch.Tensor]=None) -> torch.Tensor:
        try:
            with torch.no_grad():
                if target_labels is not None:
                    labels = target_labels.to(self.device).long()
                else:
                    try:
                        labels = model(original).argmax(dim=1).long()
                    except Exception:
                        labels = torch.zeros(original.size(0), dtype=torch.long, device=self.device)
            model_params = [p for p in model.parameters() if p.requires_grad]
            if not model_params:
                return torch.zeros((), device=self.device, requires_grad=True)
            with torch.enable_grad():
                loss_orig = F.cross_entropy(model(original.detach()), labels)
                grad_orig = torch.autograd.grad(loss_orig, model_params, create_graph=False, allow_unused=True)
            grad_orig = [g.detach() if g is not None else None for g in grad_orig]
            loss_cand = F.cross_entropy(model(candidate), labels)
            grad_cand = torch.autograd.grad(loss_cand, model_params, create_graph=bool(self.config.gradient_second_order), retain_graph=True, allow_unused=True)
            total, n_valid = (None, 0)
            flat_orig, flat_cand = ([], [])
            for g1, g2 in zip(grad_orig, grad_cand):
                if g1 is None or g2 is None:
                    continue
                flat_orig.append(g1.reshape(-1))
                flat_cand.append(g2.reshape(-1))
                n_valid += 1
            if n_valid == 0:
                return torch.zeros((), device=self.device, requires_grad=True)
            g_o = torch.cat(flat_orig)
            g_c = torch.cat(flat_cand)
            cos = F.cosine_similarity(g_o.unsqueeze(0), g_c.unsqueeze(0), dim=-1).mean()
            direction = 1.0 - cos
            relative = (g_c - g_o).norm() / (g_o.norm() + 1e-08)
            return direction + float(self.config.gradient_relative_weight) * relative
        except Exception as e:
            logger.warning(f'Gradient utility loss failed: {e}')
            return torch.zeros((), device=self.device, requires_grad=True)

    def _structure_preservation_loss(self, original: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        try:
            if self.bio_type == 'ECG':
                return self._ecg_structure_loss(original, candidate)
            elif self.bio_type == 'PPG':
                return self._ppg_structure_loss(original, candidate)
            elif self.bio_type == 'EEG':
                return self._eeg_structure_loss(original, candidate)
            else:
                mean_loss = F.mse_loss(original.mean(), candidate.mean())
                std_loss = F.mse_loss(original.std(), candidate.std())
                return mean_loss + std_loss
        except Exception as e:
            logger.warning(f'Structure preservation loss failed: {e}')
            return torch.tensor(0.0, device=self.device)

    def _ecg_structure_loss(self, original: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        try:
            w = getattr(self.config, 'structure_block_weights', None)
            o = original.view(original.size(0), -1)
            c = candidate.view(candidate.size(0), -1)
            if not w:
                return 1.0 - F.cosine_similarity(o, c, dim=1).mean()
            L = o.size(-1)
            num = None
            den = 0.0
            for (name, spans), wk in zip(ECG_BLOCK_SPEC, w):
                wk = float(wk)
                if wk <= 0:
                    continue
                idx = torch.cat([torch.arange(s, min(e, L), device=o.device) for s, e in spans if min(e, L) > s]) if spans else torch.empty(0, dtype=torch.long, device=o.device)
                if idx.numel() < 2:
                    continue
                cos_k = F.cosine_similarity(o[:, idx], c[:, idx], dim=1).mean()
                term = wk * (1.0 - cos_k)
                num = term if num is None else num + term
                den += wk
            if num is None or den <= 0:
                return 1.0 - F.cosine_similarity(o, c, dim=1).mean()
            return num / den
        except Exception as e:
            logger.warning(f'ECG structure loss failed: {e}')
            return torch.tensor(0.0, device=self.device)

    def _ppg_structure_loss(self, original: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        try:
            return F.mse_loss(original, candidate) + F.mse_loss(torch.diff(original), torch.diff(candidate))
        except Exception as e:
            logger.warning(f'PPG structure loss failed: {e}')
            return torch.tensor(0.0, device=self.device)

    def _eeg_structure_loss(self, original: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        try:
            orig_fft = torch.fft.fft(original.detach())
            cand_fft = torch.fft.fft(candidate.detach())
            return F.mse_loss(torch.abs(orig_fft), torch.abs(cand_fft))
        except Exception as e:
            logger.warning(f'EEG structure loss failed: {e}')
            return torch.tensor(0.0, device=self.device)

    def _semantic_privacy_loss(self, original: torch.Tensor, candidate: torch.Tensor, guidance: Optional[torch.Tensor]=None, semantic_texts: Optional[List[str]]=None) -> torch.Tensor:
        try:
            if self.config.semantic_source == 'stat_feature' and self.stat_encoder is not None:
                target = self.stat_encoder(original)
            elif self.config.use_learnable_text_encoder and semantic_texts:
                target = self.text_encoder(semantic_texts, device=self.device)
            elif guidance is not None:
                target = guidance.to(self.device)
            else:
                return torch.zeros((), device=self.device, requires_grad=True)
            if target.dim() == 1:
                target = target.unsqueeze(0)
            if target.size(0) != candidate.size(0):
                if target.size(0) < candidate.size(0):
                    pad = target[-1:].expand(candidate.size(0) - target.size(0), -1)
                    target = torch.cat([target, pad], dim=0)
                else:
                    target = target[:candidate.size(0)]
            target = F.normalize(target, p=2, dim=-1).detach()
            cand_feat = self.generator.encode(candidate)
            pred_rob = F.normalize(self.w_proj(cand_feat), p=2, dim=-1)
            with torch.no_grad():
                raw_feat = self.generator.encode(original.detach())
                pred_raw = F.normalize(self.w_proj(raw_feat), p=2, dim=-1).detach()
            align = 1.0 - F.cosine_similarity(pred_rob, target, dim=-1)
            away = F.relu(float(self.config.semantic_margin) - (1.0 - F.cosine_similarity(pred_rob, pred_raw, dim=-1)))
            return (align + away).mean()
        except Exception as e:
            logger.warning(f'Semantic privacy loss failed: {e}')
            return torch.zeros((), device=self.device, requires_grad=True)

    def _resolve_visual_margin(self, model: nn.Module) -> float:
        if self.config.visual_margin_mode == 'absolute':
            return float(self.config.visual_margin)
        c = getattr(model, 'num_classes', None)
        if not isinstance(c, int) or c < 2:
            c = self._infer_num_classes(model)
        if not isinstance(c, int) or c < 2:
            return float(self.config.visual_margin)
        return float(self.config.visual_margin_ratio) * float(np.log(c))

    def _infer_num_classes(self, model: nn.Module) -> Optional[int]:
        if self._num_classes_hint is not None:
            return self._num_classes_hint
        return None

    def _visual_privacy_loss(self, original: torch.Tensor, candidate: torch.Tensor, model: nn.Module) -> torch.Tensor:
        try:
            with torch.no_grad():
                logits_orig = model(original.detach())
                prob_orig = F.softmax(logits_orig, dim=1)
                self._num_classes_hint = int(logits_orig.size(1))
                if not bool(self.config.visual_detach_target):
                    prob_orig = prob_orig.detach()
            log_prob_cand = F.log_softmax(model(candidate), dim=1)
            visual_loss = F.kl_div(log_prob_cand, prob_orig, reduction='batchmean')
            return visual_loss
        except Exception as e:
            logger.warning(f'Visual privacy loss failed: {e}')
            return torch.zeros((), device=self.device, requires_grad=True)

    def _perturbation_constraint(self, original: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
        try:
            perturbation = candidate - original
            perturbation_norm = torch.norm(perturbation, p=2)
            original_norm = torch.norm(original, p=2)
            if original_norm > 0:
                return perturbation_norm / original_norm
            return perturbation_norm
        except Exception as e:
            logger.warning(f'Perturbation constraint failed: {e}')
            return torch.tensor(0.0, device=self.device)

    def _postprocess_robust_data(self, original: torch.Tensor, robust: torch.Tensor) -> torch.Tensor:
        try:
            with torch.no_grad():
                denom = float(original.norm()) + 1e-12
                self.last_perturbation_before_post = float((robust - original).norm() / denom)
            if torch.allclose(robust, robust[0]):
                self.last_postprocess_fallback = True
                return original.clone()
            robust = torch.clamp(robust, -5.0, 5.0)
            if self.config.postprocess_match_stats:
                with torch.no_grad():
                    orig_mean, orig_std = (original.mean(), original.std())
                    rob_mean, rob_std = (robust.mean(), robust.std())
                    if rob_std > 0 and orig_std > 0:
                        robust_norm = (robust - rob_mean) / rob_std
                        robust = robust_norm * orig_std + orig_mean
            with torch.no_grad():
                denom = float(original.norm()) + 1e-12
                self.last_perturbation_after_post = float((robust - original).norm() / denom)
            return robust
        except Exception as e:
            return original.clone()

    def batch_generate(self, data_batch: torch.Tensor, model: nn.Module, current_round: int=0) -> torch.Tensor:
        try:
            robust_batch = []
            for i in range(data_batch.size(0)):
                single_sample = data_batch[i:i + 1]
                robust_sample = self.generate_robust_data(single_sample, model, current_round=current_round)
                robust_batch.append(robust_sample)
            return torch.cat(robust_batch, dim=0)
        except Exception as e:
            logger.error(f'Batch generation failed: {e}')
            return data_batch.clone()

    def get_generation_report(self) -> Dict[str, Any]:
        try:
            return {'total_generations': len(self.generation_history['losses']), 'avg_loss': np.mean(self.generation_history['losses']) if self.generation_history['losses'] else 0.0, 'avg_semantic_score': np.mean(self.generation_history['semantic_scores']) if self.generation_history['semantic_scores'] else 0.0, 'avg_structure_score': np.mean(self.generation_history['structure_scores']) if self.generation_history['structure_scores'] else 0.0, 'avg_visual_score': np.mean(self.generation_history['visual_scores']) if self.generation_history['visual_scores'] else 0.0, 'history': self.generation_history}
        except Exception as e:
            logger.error(f'Generation report failed: {e}')
            return {'error': str(e)}

    def save_state(self, save_path: Path):
        try:
            state = {'generator_state_dict': self.generator.state_dict(), 'optimizer_state_dict': self.optimizer.state_dict(), 'generation_history': self.generation_history, 'generation_config': self.config}
            torch.save(state, save_path)
            logger.info(f'LBRD Generator state saved to {save_path}')
        except Exception as e:
            logger.error(f'State save failed: {e}')

    def load_state(self, load_path: Path):
        try:
            if load_path.exists():
                state = torch.load(load_path, map_location=self.device)
                self.generator.load_state_dict(state['generator_state_dict'])
                self.optimizer.load_state_dict(state['optimizer_state_dict'])
                self.generation_history = state['generation_history']
                logger.info(f'LBRD Generator state loaded from {load_path}')
        except Exception as e:
            logger.error(f'State load failed: {e}')

    def get_stats(self) -> Dict[str, Any]:
        try:
            return {'bio_type': self.bio_type, 'total_generations': len(self.generation_history['losses']), 'avg_generation_loss': np.mean(self.generation_history['losses']) if self.generation_history['losses'] else 0.0, 'config': self.config}
        except Exception as e:
            logger.error(f'Stats generation failed: {e}')
            return {'error': str(e)}

def create_lbrd_generator(llm_guide, bio_type: str='ECG', device: str='cuda', config: LBRDConfig=None) -> LBRDGenerator:
    return LBRDGenerator(llm_guide, bio_type, device, config)