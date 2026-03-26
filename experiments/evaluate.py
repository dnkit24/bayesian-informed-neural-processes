"""Evaluate trained models across knowledge perturbation levels and context sizes.

Usage (from code_repo_meta/):
    conda run -n cam_env python experiments/evaluate.py
"""

import torch
import numpy as np
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))

from config import Config
from dataset.utils import get_dataloader
from dataset.dataset import SetKnowledgeTrendingSinusoids
from models.inp import INP
from models.binp import BINP
from models.loss import sum_log_prob

from experiments.train import SEED, MODEL_SPECS, make_config, find_model_dir

DELTAS = [0, 0.5, 1.0, 2.0, 3.0, 4.0]
N_CONTEXTS = [0, 1, 3, 5, 10]
N_CONTEXTS_EXTENDED = [0, 1, 3, 5, 10, 15, 20]
MAX_EVAL_BATCHES = 40


def load_trained_model(save_dir, checkpoint="model_best.pt"):
    """Load a trained model from checkpoint directory."""
    config = Config.from_toml(os.path.join(save_dir, "config.toml"))
    config.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    config.test_num_z_samples = 32
    config.knowledge_dropout = 0

    model_type = getattr(config, "model_type", "inp")
    if model_type.startswith("binp"):
        model = BINP(config)
    else:
        model = INP(config)

    model.to(config.device)
    ckpt_path = os.path.join(save_dir, checkpoint)
    state_dict = torch.load(ckpt_path, map_location=config.device)
    model.load_state_dict(state_dict)
    model.eval()
    return model, config


def load_all_models(project_name="binp_project"):
    """Load all trained models. Returns dict of {name: (model, config)}."""
    models = {}
    for model_name, _, prefix in MODEL_SPECS:
        save_dir = find_model_dir(project_name, prefix)
        if save_dir is None:
            print(f"[WARN] No checkpoint found for {model_name}")
            continue
        checkpoint = "model_v3.pt" if model_name.startswith("BINP") else "model_best.pt"
        print(f"Loading {model_name} from {save_dir} ({checkpoint})")
        model, config = load_trained_model(save_dir, checkpoint=checkpoint)
        models[model_name] = (model, config)
    return models


def perturb_knowledge(knowledge, param_idx, delta):
    """Perturb knowledge by adding delta to a specific parameter's value.

    For sinusoids, knowledge shape is [bs, 3, 4] where columns are
    [one-hot indicator (3 cols), value (1 col)].
    """
    knowledge = knowledge.clone()
    knowledge[:, param_idx, -1] += delta
    return knowledge


def _sample_context(x_target, y_target, n_ctx, rng):
    """Deterministically sample context points using a seeded RNG."""
    if n_ctx > 0 and x_target.shape[1] > n_ctx:
        ctx_idx = rng.choice(x_target.shape[1], n_ctx, replace=False)
        return x_target[:, ctx_idx, :], y_target[:, ctx_idx, :]
    return x_target[:, :0, :], y_target[:, :0, :]


def evaluate_model(model, dataloader, deltas, n_contexts, config, is_binp=False,
                   param_idx=1):
    """Evaluate a model across perturbation levels and context sizes.

    Returns dict with keys:
        "nll":         {(delta, n_ctx): mean_nll}
        "nll_se":      {(delta, n_ctx): standard_error}
        "diagnostics": {(delta, n_ctx): {sigma_D, sigma_K, variance_ratio,
                                          variance_ratio_se}} (BINP only)
    """
    device = config.device
    model.eval()

    results = {"nll": {}, "nll_se": {}, "diagnostics": {}}

    for delta in deltas:
        for n_ctx in n_contexts:
            eval_rng = np.random.RandomState(SEED + n_ctx)
            nll_values = []
            diag_sigma_D = []
            diag_sigma_K = []
            diag_variance_ratio = []

            batch_count = 0
            for batch in dataloader:
                if batch_count >= MAX_EVAL_BATCHES:
                    break
                batch_count += 1

                _, target, knowledge, _ = batch
                x_target, y_target = target
                x_target = x_target.to(device)
                y_target = y_target.to(device)

                x_context, y_context = _sample_context(
                    x_target, y_target, n_ctx, eval_rng,
                )

                if config.use_knowledge and knowledge is not None:
                    k_perturbed = perturb_knowledge(knowledge, param_idx=param_idx, delta=delta)
                else:
                    k_perturbed = None

                with torch.no_grad():
                    k_arg = k_perturbed if (config.use_knowledge and k_perturbed is not None) else None
                    outputs = model(
                        x_context, y_context, x_target,
                        y_target=None, knowledge=k_arg,
                    )

                    p_yCc = outputs[0]
                    sum_log_p = sum_log_prob(p_yCc, y_target)
                    log_S = torch.logsumexp(sum_log_p, 0)
                    log_E = log_S - math.log(sum_log_p.shape[0])
                    nll = -log_E
                    nll_values.append(nll.mean().item())

                    if is_binp:
                        diags = model.get_diagnostics()
                        if diags.get("sigma_D") is not None:
                            diag_sigma_D.append(diags["sigma_D"].mean().item())
                        if diags.get("sigma_K") is not None:
                            diag_sigma_K.append(diags["sigma_K"].mean().item())
                        if diags.get("variance_ratio") is not None:
                            diag_variance_ratio.append(diags["variance_ratio"].mean().item())

            key = (delta, n_ctx)
            nll_arr = np.array(nll_values) if nll_values else np.array([float("nan")])
            results["nll"][key] = nll_arr.mean()
            results["nll_se"][key] = nll_arr.std() / max(np.sqrt(len(nll_arr)), 1)

            if is_binp and diag_variance_ratio:
                vr_arr = np.array(diag_variance_ratio)
                results["diagnostics"][key] = {
                    "sigma_D": np.mean(diag_sigma_D),
                    "sigma_K": np.mean(diag_sigma_K),
                    "variance_ratio": vr_arr.mean(),
                    "variance_ratio_se": vr_arr.std() / max(np.sqrt(len(vr_arr)), 1),
                }

    return results


if __name__ == "__main__":
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    test_dataset = SetKnowledgeTrendingSinusoids(split="test", knowledge_type="full")
    cfg = make_config()
    cfg.knowledge_input_dim = test_dataset.knowledge_input_dim
    test_loader = get_dataloader(test_dataset, cfg)

    models = load_all_models()

    print("\n" + "=" * 70)
    print("NLL Results (n=5)")
    print("=" * 70)
    header = f"{'Model':<12}" + "".join(f"{'d='+str(d):<14}" for d in DELTAS)
    print(header)
    print("-" * 70)

    for name, (model, config) in models.items():
        is_binp = name.startswith("BINP")
        n_ctxs = N_CONTEXTS_EXTENDED if is_binp else N_CONTEXTS
        results = evaluate_model(model, test_loader, DELTAS, n_ctxs, config, is_binp=is_binp)
        row = f"{name:<12}"
        for d in DELTAS:
            val = results["nll"].get((d, 5), float("nan"))
            se = results["nll_se"].get((d, 5), 0)
            row += f"{val:>8.1f}({se:.1f})  "
        print(row)

        if is_binp and results["diagnostics"]:
            print(f"  VR (n=5): ", end="")
            for d in DELTAS:
                diag = results["diagnostics"].get((d, 5))
                if diag:
                    print(f"d={d}: {diag['variance_ratio']:.3f}  ", end="")
            print()
