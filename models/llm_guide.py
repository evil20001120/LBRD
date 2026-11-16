# Reference implementation for the LBRD-PPFL manuscript.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import requests
import json
import time
from pathlib import Path
import pickle
import warnings
from dataclasses import dataclass, field
import hashlib
import os
try:
    from .semantic_encoder import SemanticTextEncoder, TextEncoderConfig
except ImportError:
    from semantic_encoder import SemanticTextEncoder, TextEncoderConfig
warnings.filterwarnings('ignore')

def _load_api_key() -> str:
    key = os.environ.get('DEEPSEEK_API_KEY', '').strip()
    if key:
        return key
    for _p in (Path(__file__).resolve().parents[1] / '.secrets' / 'deepseek.key', Path.home() / '.deepseek_key'):
        try:
            if _p.exists():
                _k = _p.read_text(encoding='utf-8').strip()
                if _k:
                    return _k
        except Exception:
            pass
    return ''

@dataclass
class DeepSeekConfig:
    model_name: str = 'deepseek-chat'
    api_key: str = field(default_factory=_load_api_key)
    base_url: str = 'https://api.deepseek.com'
    max_length: int = 512
    temperature: float = 0.3
    timeout: int = 30
    max_retries: int = 3
    rate_limit: int = 60
    cache_enabled: bool = True
    cache_size: int = 1000

class LLMGuide:

    def __init__(self, bio_type: str='ECG', config: DeepSeekConfig=None):
        self.bio_type = bio_type.upper()
        self.config = config or DeepSeekConfig()
        self.client = None
        self._init_client()
        self.text_encoder: Optional[SemanticTextEncoder] = None
        self._text_cache: Dict[str, str] = {}
        self.embedding_cache = {}
        self.cache_size = 1000
        self.cache_enabled = self.config.cache_enabled
        self.prompt_templates = self._init_prompt_templates()
        self.api_calls = 0
        self.cache_hits = 0
        self.total_latency = 0
        print(f'LLMGuide initialized for {self.bio_type} with DeepSeek Chat API')

    def _init_client(self):
        try:
            if not self.config.api_key:
                print('⚠️  System will use fallback random embeddings')
                self.client = None
                return
            import openai
            self.client = openai.OpenAI(api_key=self.config.api_key, base_url=self.config.base_url)
            self._test_connection()
            print('DeepSeek API client initialized successfully')
        except Exception as e:
            print(f'⚠️  Warning: DeepSeek API client initialization failed: {e}')
            print('⚠️  System will use fallback random embeddings')
            self.client = None

    def _test_connection(self):
        try:
            response = self.client.chat.completions.create(model=self.config.model_name, messages=[{'role': 'user', 'content': 'Hello'}], max_tokens=5)
            print('✅ DeepSeek API connection test passed')
        except Exception as e:
            print(f'❌ API connection test failed: {e}')
            raise

    def _init_prompt_templates(self) -> Dict[str, Dict]:
        return {'ECG': {'description': '', 'features': ['', '', '', '', '', '', ''], 'context': ''}, 'PPG': {'description': '', 'features': ['', '', '', '', ''], 'context': ''}, 'EEG': {'description': '', 'features': ['', '', '', '', '', ''], 'context': ''}}

    def extract_semantic_embedding(self, data: torch.Tensor, use_cache: bool=True) -> torch.Tensor:
        if self.client is None:
            return torch.randn(data.size(0), 768, device=data.device)
        if use_cache and self.cache_enabled:
            cache_key = self._generate_cache_key(data)
            if cache_key in self.embedding_cache:
                self.cache_hits += 1
                return self.embedding_cache[cache_key].to(data.device)
        start_time = time.time()
        try:
            embeddings = []
            batch_size = min(data.size(0), 5)
            for i in range(batch_size):
                sample_data = data[i].squeeze().detach().cpu().numpy()
                text_description = self._generate_description(sample_data)
                semantic_description = self._call_chat_api(text_description)
                embedding = self._text_to_embedding(semantic_description)
                embeddings.append(embedding)
            while len(embeddings) < data.size(0):
                embeddings.append(embeddings[-1])
            embedding_tensor = torch.stack(embeddings).to(data.device)
            if use_cache and self.cache_enabled:
                self._add_to_cache(cache_key, embedding_tensor.cpu())
            self.api_calls += 1
            self.total_latency += time.time() - start_time
            return embedding_tensor
        except Exception as e:
            print(f'⚠️  Semantic extraction failed: {e}, using random fallback')
            return torch.randn(data.size(0), 768, device=data.device)

    def _generate_description(self, signal: np.ndarray) -> str:
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        min_val = np.min(signal)
        max_val = np.max(signal)
        fft = np.fft.fft(signal)
        freqs = np.fft.fftfreq(len(signal))
        dominant_freq = abs(freqs[np.argmax(np.abs(fft))])
        template = self.prompt_templates.get(self.bio_type, self.prompt_templates['ECG'])
        description = f"{template['description']}:\n"
        description += f'{mean_val:.3f}{std_val:.3f}{max_val:.3f}{min_val:.3f}\n'
        description += f'{dominant_freq:.3f} Hz\n'
        description += f'{len(signal)}'
        description += f"{template['context']}\n"
        description += ''
        return description

    def _call_chat_api(self, prompt: str) -> str:
        if self.client is None:
            return f'[offline-fallback] {prompt}'
        max_retries = self.config.max_retries
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(model=self.config.model_name, messages=[{'role': 'system', 'content': f'{self.bio_type}'}, {'role': 'user', 'content': prompt}], temperature=self.config.temperature, max_tokens=300)
                return response.choices[0].message.content
            except Exception as e:
                print(f'API call failed (attempt {attempt + 1}/{max_retries}): {e}')
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return f'{self.bio_type}{len(prompt)}'

    def set_text_encoder(self, encoder: Optional[SemanticTextEncoder]) -> None:
        self.text_encoder = encoder

    @staticmethod
    def _fallback_embedding(text: str, dim: int=768) -> torch.Tensor:
        raw = np.frombuffer(hashlib.sha256(text.encode('utf-8')).digest(), dtype=np.uint8)
        repeats = dim // raw.size + 1
        vec = np.tile(raw.astype(np.float32), repeats)[:dim]
        vec = vec / 127.5 - 1.0
        norm = float(np.linalg.norm(vec))
        if norm <= 0:
            vec = np.ones(dim, dtype=np.float32) / np.sqrt(dim)
        else:
            vec = vec / norm
        return torch.tensor(vec, dtype=torch.float32)

    def get_semantic_texts(self, data: torch.Tensor, max_batch: int=5) -> List[str]:
        texts: List[str] = []
        n = min(int(data.size(0)), max_batch)
        for i in range(n):
            sample = data[i].squeeze().detach().cpu().numpy()
            desc = self._generate_description(sample)
            key = hashlib.md5(desc.encode('utf-8')).hexdigest()
            if key in self._text_cache:
                texts.append(self._text_cache[key])
                continue
            try:
                txt = self._call_chat_api(desc)
            except Exception:
                txt = ''
            self._text_cache[key] = txt
            texts.append(txt)
        if not texts:
            texts = ['']
        while len(texts) < int(data.size(0)):
            texts.append(texts[-1])
        return texts

    def _text_to_embedding(self, text: str) -> torch.Tensor:
        if self.text_encoder is not None:
            with torch.no_grad():
                return self.text_encoder([text]).squeeze(0).detach()
        return self._fallback_embedding(text)

    def compute_semantic_distance(self, embedding1: torch.Tensor, embedding2: torch.Tensor, metric: str='cosine') -> float:
        if metric == 'cosine':
            similarity = F.cosine_similarity(embedding1.view(1, -1), embedding2.view(1, -1))
            distance = 1.0 - similarity.item()
        elif metric == 'euclidean':
            distance = torch.norm(embedding1 - embedding2, p=2).item()
            distance = min(distance / 10.0, 1.0)
        else:
            similarity = F.cosine_similarity(embedding1.view(1, -1), embedding2.view(1, -1))
            distance = 1.0 - similarity.item()
        return max(0.0, min(1.0, distance))

    def generate_semantic_guidance(self, original_data: torch.Tensor, target_class: int=None, guidance_strength: float=0.5) -> torch.Tensor:
        if self.client is None:
            return torch.randn(original_data.size(0), 768, device=original_data.device) * 0.1
        orig_embedding = self.extract_semantic_embedding(original_data, use_cache=True)
        if target_class is not None:
            target_description = self._get_target_description(target_class)
            target_embedding = self._text_to_embedding(target_description)
            target_embedding = target_embedding.to(orig_embedding.device).unsqueeze(0)
        else:
            target_embedding = orig_embedding + torch.randn_like(orig_embedding) * 0.3
        guidance = (1 - guidance_strength) * orig_embedding + guidance_strength * target_embedding
        return guidance

    def _get_target_description(self, target_class: int) -> str:
        descriptions = {0: '', 1: f'{self.bio_type}', 2: f'{self.bio_type}', 3: f'{self.bio_type}', 4: f'{self.bio_type}'}
        return descriptions.get(target_class, '')

    def _generate_cache_key(self, data: torch.Tensor) -> str:
        import hashlib
        data_bytes = data.detach().cpu().numpy().tobytes()
        return hashlib.md5(data_bytes).hexdigest()

    def _add_to_cache(self, key: str, embedding: torch.Tensor):
        if len(self.embedding_cache) >= self.cache_size:
            oldest_key = next(iter(self.embedding_cache))
            del self.embedding_cache[oldest_key]
        if embedding.requires_grad:
            embedding = embedding.detach()
        self.embedding_cache[key] = embedding.cpu()

    def get_stats(self) -> Dict[str, Any]:
        avg_latency = self.total_latency / self.api_calls if self.api_calls > 0 else 0
        return {'api_calls': self.api_calls, 'cache_hits': self.cache_hits, 'cache_size': len(self.embedding_cache), 'avg_latency_ms': avg_latency * 1000, 'hit_rate': self.cache_hits / (self.cache_hits + self.api_calls) if self.cache_hits + self.api_calls > 0 else 0, 'bio_type': self.bio_type, 'api_available': self.client is not None}

    def save_cache(self, cache_path: Path):
        if not self.cache_enabled:
            return
        with open(cache_path, 'wb') as f:
            pickle.dump(self.embedding_cache, f)
        print(f'✅ Embedding cache saved to {cache_path}')

    def load_cache(self, cache_path: Path):
        if not self.cache_enabled:
            return
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                self.embedding_cache = pickle.load(f)
            print(f'✅ Embedding cache loaded from {cache_path}, {len(self.embedding_cache)} entries')

    def clear_cache(self):
        self.embedding_cache.clear()
        print('✅ Embedding cache cleared')

def create_llm_guide(bio_type: str='ECG', config: DeepSeekConfig=None) -> LLMGuide:
    return LLMGuide(bio_type, config)