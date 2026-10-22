# Bayesian Informed Neural Processes

This repository extends [Informed Neural Processes](https://github.com/kasia-kobalczyk/informed-meta-learning) (Kobalczyk & van der Schaar, ICLR 2025) with precision-weighted Bayesian knowledge fusion. The standard INP integrates domain knowledge via additive fusion, which cannot suppress wrong knowledge regardless of how much data is observed. BINP replaces this with a Gaussian product over separate data and knowledge belief distributions, enabling learned, per-dimension reliability weighting.

## Paper

[**Learning Source Reliability in Informed Neural Processes**](paper/learning-source-reliability.pdf) — David Nelischer.

The accompanying paper describes the method, theoretical analysis, and experiments on synthetic and real-world temperature data.

## Method

BINP introduces separate data and knowledge encoders that produce independent Gaussian beliefs over the latent space:

```
q_D = N(mu_D, sigma_D^2)    from data encoder MLP_D
q_K = N(mu_K, sigma_K^2)    from knowledge encoder MLP_K
```

A disagreement mechanism inflates `sigma_K` when the two beliefs diverge, reducing the influence of unreliable knowledge. The beliefs are combined via precision-weighted Gaussian product:

```
prec_* = 1/sigma_D^2 + 1/sigma_K^2
mu_*   = (mu_D/sigma_D^2 + mu_K/sigma_K^2) / prec_*
```

Training uses a two-phase protocol: Phase 1 (300 epochs) learns correct-knowledge representations with full learning rate; Phase 2 (1700 epochs) introduces knowledge corruption and differential learning rates to calibrate the reliability mechanism.

## Setup

```bash
conda env create -f environment.yaml
conda activate inps
```

## Training

Train all four models (NP, INP, INP_c, BINP_c) sequentially:

```bash
python experiments/train.py
```

Models are saved to `saves/binp_project/`. BINP_c and INP_c are automatically initialized from the trained INP checkpoint.

## Evaluation

Evaluate across knowledge perturbation levels and context sizes:

```bash
python experiments/evaluate.py
```

## Results

NLL on trending sinusoids (n=5 context points, 32 z-samples, 40 batches). Delta controls knowledge misspecification (0 = correct, higher = more wrong).

| delta | NP | INP | INP(c) | **BINP(c)** |
|------:|-----:|------:|-------:|--------:|
| 0   | 71.9 | **-16.3** | -13.1 | -10.2 |
| 0.5 | 67.8 | 55.2  | 8.1   | **3.9** |
| 1.0 | 65.9 | 315.6 | 93.3  | **63.1** |
| 2.0 | 69.0 | 948.5 | 320.2 | **202.6** |
| 3.0 | 73.1 | 833.3 | 378.2 | **159.7** |
| 4.0 | 68.9 | 517.5 | 359.0 | **102.5** |

BINP(c) dominates under severe misspecification (delta >= 1) while remaining comparable at delta = 0.

## Acknowledgments

This work builds on the [Informed Meta-Learning](https://github.com/kasia-kobalczyk/informed-meta-learning) codebase by Kobalczyk & van der Schaar (ICLR 2025). The base INP model, dataset utilities, and training infrastructure are from the original repository.
