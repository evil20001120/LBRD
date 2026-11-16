# LBRD-PPFL: Privacy-Preserving Federated Learning for Biometric Data with LLM-Assisted Semantic Data Synthesis

Reference implementation for the manuscript:

> LBRD-PPFL: A Privacy-Preserving Federated Learning Framework for Biometric Data with LLM-Assisted Semantic Data Synthesis

The repository contains the LBRD robust-data generator, the federated training and
evaluation pipeline used for the ECG / PPG / EEG benchmarks, the membership-inference
evaluation, and the cached LLM semantic descriptions used by the generator.

## Contents

```
models/
  lbrd_generator.py       LBRD generator: four-objective optimization
                          (gradient utility, structure preservation, semantic
                          alignment + margin-bounded separation, visual privacy),
                          sensitivity-weighted block-wise structure loss,
                          DP noise convention (sigma*C/sqrt(d))
  llm_guide.py            LLM semantic extraction client (DeepSeek endpoint),
                          deterministic hash-based embedding, FIFO cache,
                          offline fallback
  semantic_encoder.py     Semantic encoder variants (frozen hash encoder,
                          statistics-prior init, learnable InfoNCE encoder,
                          statistical-features branch)
13_prepare_ecg_real.py   MIT-BIH windowing, record-to-pathology-group mapping,
                          class-balanced federated partitioning
14_make_real_robust.py   Offline robust-corpus generation for ECG/PPG/EEG
12_rerun_real.py         Federated training + DP pipeline
                          (FedAvg / FedAvg+DP / LBRD / LBRD+DP / FedProx / DP-FedCGAN),
                          replace-protocol data merging, RDP accountant
15_mia_real.py           Loss-threshold membership inference evaluation
                          (advantage, ROC-AUC, TPR at fixed FPR budgets)
README.md                This file
requirements.txt         Python dependencies
```

## Environment

- Python 3.10, PyTorch 1.13.1 (CUDA), scikit-learn, numpy
- Public datasets: MIT-BIH Arrhythmia Database, CHB-MIT, PPG-DaLiA (PhysioNet)

## Notes

- The DeepSeek API key is read from the environment variable `DEEPSEEK_API_KEY`;
  no credentials are stored in this repository. All reported experiments use the
  cached semantic descriptions, so no API access is required to reproduce them.
- The DP mechanism injects noise per parameter element with standard deviation
  `sigma*C/sqrt(d)` (total noise norm `sigma*C` against the clipped update norm);
  the relation between this convention and the nominal RDP calibration target is
  analyzed in Section 6.2 and Appendix A of the manuscript, and no formal
  (epsilon, delta) guarantee is claimed for the implemented mechanism.
- Under the replacement protocol, every training window is replaced by its robust
  surrogate; the membership-inference members are these replaced source windows.

## Cached corpora

Two archives (attached in the repository root, each below the 100 MB per-file limit)
contain the pre-generated robust corpora and the cached LLM semantic descriptions
used in the reported experiments:

- `cache-corpora.zip` — the robust surrogate corpora (`real_robust/`, `real_robust_npz/`)
  and the generated EEG and PPG datasets (`EEG_v2.0/`, `PPG_v2.0/`) read by the
  training and evaluation scripts;
- `cache-llm-descriptions.zip` — the dated ECG semantic-description caches produced
  with the LLM channel (`ECG_v2.0_*`).

Extract both next to the scripts so that a `lbrd_cache/` directory is formed;
the pipelines then run fully offline and no API access is needed. Intermediate
exploration-era cache versions are not included.
