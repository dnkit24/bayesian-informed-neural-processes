"""Train all models (NP, INP, INP_c, BINP_c) on trending sinusoids.

Usage (from code_repo_meta/):
    conda run -n cam_env python experiments/train.py
"""

import torch
import numpy as np
import os
import sys
import re
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))

from config import Config
from models.train_binp import meta_train_binp

SEED = 42
SINUSOID_DEFAULTS = dict(
    dataset="set-trending-sinusoids",
    knowledge_type="full",
    batch_size=64,
    lr=1e-3,
    num_epochs=2000,
    noise=0.2,
    hidden_dim=128,
    knowledge_dim=128,
    text_encoder="set",
    train_num_z_samples=1,
    test_num_z_samples=16,
    min_num_context=0,
    max_num_context=10,
    num_targets=100,
    latent_encoder_num_hidden=2,
    knowledge_extractor_num_hidden=0,
    seed=SEED,
    input_dim=1,
    output_dim=1,
    xy_encoder_num_hidden=2,
    xy_encoder_hidden_dim=384,
    data_agg_func="mean",
    decoder_hidden_dim=128,
    decoder_num_hidden=3,
    decoder_activation="gelu",
    x_transf_dim=128,
    x_encoder_num_hidden=1,
    sort_context=False,
    x_sampler="uniform",
    knowledge_dropout=0,
    beta=1.0,
    n_trials=1,
    freeze_llm=True,
    tune_llm_layer_norms=False,
    knowledge_extractor_hidden_dim=128,
)


def make_config(**overrides):
    """Create a Config with sinusoid defaults and any overrides applied."""
    cfg = dict(SINUSOID_DEFAULTS)
    cfg.update(overrides)
    cfg.setdefault("run_name_prefix", "run")
    cfg.setdefault("model_type", "inp")
    cfg.setdefault("use_knowledge", True)
    cfg.setdefault("knowledge_merge", "sum")
    cfg.setdefault("p_shuffle", 0.0)
    cfg.setdefault("p_absent", 0.0)
    cfg.setdefault("early_stopping_patience", 0)
    cfg.setdefault("project_name", "binp_project")
    use_gpu = torch.cuda.is_available()
    cfg["device"] = torch.device("cuda:0") if use_gpu else torch.device("cpu")
    return Config(**cfg)


MODEL_SPECS = [
    (
        "NP",
        dict(use_knowledge=False, knowledge_dropout=0, run_name_prefix="np"),
        "np",
    ),
    (
        "INP",
        dict(
            use_knowledge=True,
            knowledge_dropout=0.3,
            knowledge_merge="sum",
            run_name_prefix="inp",
        ),
        "inp",
    ),
    (
        "BINP_c",
        dict(
            use_knowledge=True,
            knowledge_dropout=0,
            model_type="binp_twophase",
            p_shuffle=0.10,
            p_absent=0.05,
            phase2_epoch=300,
            phase2_lr_factor=0.2,
            disagree_bias_init=-1.5,
            disagree_scale_init=1.5,
            num_epochs=2000,
            early_stopping_patience=40,
            run_name_prefix="binp_c",
        ),
        "binp_c",
    ),
    (
        "INP_c",
        dict(
            use_knowledge=True,
            knowledge_dropout=0,
            knowledge_merge="sum",
            model_type="inp_c",
            p_shuffle=0.10,
            p_absent=0.05,
            num_epochs=2000,
            early_stopping_patience=40,
            run_name_prefix="inp_c",
        ),
        "inp_c",
    ),
]


def find_model_dir(project_name, prefix):
    """Find the latest save directory for a given model prefix.

    Uses exact prefix matching: 'inp_0' matches prefix 'inp' but
    'inp_c_0' does not (the part after prefix must be '_' + digits).
    """
    save_root = f"./saves/{project_name}"
    if not os.path.exists(save_root):
        return None
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    dirs = [(d, int(pattern.match(d).group(1))) for d in os.listdir(save_root)
            if pattern.match(d)]
    if not dirs:
        return None
    latest = max(dirs, key=lambda x: x[1])[0]
    full_path = os.path.join(save_root, latest)
    if os.path.exists(os.path.join(full_path, "model_best.pt")):
        return full_path
    return None


def train_all_models(project_name="binp_project"):
    """Train all models sequentially, saving to ./saves/{project_name}/.

    BINP_c and INP_c are initialized from the INP checkpoint
    (correct-K encoder provides a strong starting point for both).
    """
    print("=" * 60)
    print("Training all models")
    print("=" * 60)

    inp_ckpt = None

    for model_name, overrides, prefix in MODEL_SPECS:
        existing_dir = find_model_dir(project_name, prefix)
        if existing_dir:
            print(f"[SKIP] {model_name}: {existing_dir}")
            if prefix == "inp":
                inp_ckpt = os.path.join(existing_dir, "model_best.pt")
            continue

        print(f"\n{'='*60}")
        print(f"Training: {model_name}")
        print(f"{'='*60}")

        if overrides.get("model_type") in ("binp_twophase", "inp_c") and inp_ckpt:
            overrides = dict(overrides, inp_init_path=inp_ckpt)

        cfg = make_config(project_name=project_name, **overrides)

        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

        meta_train_binp(None, cfg, run_name_prefix=prefix)

        if prefix == "inp":
            d = find_model_dir(project_name, "inp")
            if d:
                inp_ckpt = os.path.join(d, "model_best.pt")

    print("\nAll models trained.")


def train_dist_shift_models(project_name="binp_dist_shift"):
    """Train all models on the distribution-shift sinusoid dataset.

    Same hyperparameters as standard sinusoids, different dataset:
    train b ~ N(2,1), test b ~ N(3,1).
    """
    dist_shift_specs = []
    for model_name, overrides, prefix in MODEL_SPECS:
        ov = dict(overrides, dataset="set-trending-sinusoids-dist-shift")
        dist_shift_specs.append((model_name, ov, prefix))

    print("=" * 60)
    print("Training all models (distribution shift)")
    print("=" * 60)

    inp_ckpt = None
    for model_name, overrides, prefix in dist_shift_specs:
        existing_dir = find_model_dir(project_name, prefix)
        if existing_dir:
            print(f"[SKIP] {model_name}: {existing_dir}")
            if prefix == "inp":
                inp_ckpt = os.path.join(existing_dir, "model_best.pt")
            continue

        print(f"\n{'='*60}")
        print(f"Training: {model_name} (dist-shift)")
        print(f"{'='*60}")

        if overrides.get("model_type") in ("binp_twophase", "inp_c") and inp_ckpt:
            overrides = dict(overrides, inp_init_path=inp_ckpt)

        cfg = make_config(project_name=project_name, **overrides)

        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

        meta_train_binp(None, cfg, run_name_prefix=prefix)

        if prefix == "inp":
            d = find_model_dir(project_name, "inp")
            if d:
                inp_ckpt = os.path.join(d, "model_best.pt")

    print("\nAll dist-shift models trained.")


if __name__ == "__main__":
    train_all_models()
