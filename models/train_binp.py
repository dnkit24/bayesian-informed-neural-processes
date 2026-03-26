"""BINP two-phase training with knowledge corruption.

Phase 1: clean data, full LR — learns correct-K representations.
Phase 2: corruption + differential LR — calibrates sigma_K for robustness.
"""

import math, re, os, sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from models.train import Trainer, get_device, EVAL_ITER
from models.binp import BINP
from models.loss import sum_log_prob

GRAD_CLIP = 1.0


def corrupt_knowledge(knowledge, p_shuffle, p_absent, training):
    """Batch-level absence + per-sample shuffle. No-op during eval."""
    if not training or knowledge is None:
        return knowledge
    if not isinstance(knowledge, torch.Tensor):
        knowledge = torch.stack(list(knowledge))
    if torch.rand(1).item() < p_absent:
        return None
    bs = knowledge.shape[0]
    if p_shuffle > 0 and bs > 1:
        mask = torch.rand(bs) < p_shuffle
        n = mask.sum().item()
        if n > 0:
            out = knowledge.clone()
            idx = mask.nonzero(as_tuple=True)[0]
            out[idx] = knowledge[(idx + torch.randint(1, bs, (n,))) % bs]
            return out
    return knowledge


def _default_sinusoid_perturb(knowledge, delta):
    """Default perturbation: shift param b in sinusoid knowledge [bs, 3, 4]."""
    k = knowledge.clone()
    k[:, 1, -1] += delta
    return k


def _eval_nll_at_delta(model, loader, delta, device, n_ctx=5, max_batches=20, perturb_fn=None):
    """Importance-weighted NLL with knowledge perturbed by delta."""
    if perturb_fn is None:
        perturb_fn = _default_sinusoid_perturb
    model.eval()
    rng = np.random.RandomState(42 + int(delta * 100))
    nlls = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            _, (x_t, y_t), knowledge, _ = batch
            x_t, y_t = x_t.to(device), y_t.to(device)
            ci = rng.choice(x_t.shape[1], n_ctx, replace=False)
            k = perturb_fn(knowledge, delta)
            out = model(x_t[:, ci], y_t[:, ci], x_t, y_target=None, knowledge=k)
            log_p = sum_log_prob(out[0], y_t)
            nlls.append(-(torch.logsumexp(log_p, 0) - math.log(log_p.shape[0])).mean().item())
    return np.mean(nlls) if nlls else float("inf")


class BINPTrainer(Trainer):
    """BINP with INP checkpoint init and gradient clipping."""

    def _create_model(self, config):
        model = BINP(config)
        inp_path = getattr(config, "inp_init_path", None)
        if inp_path and os.path.exists(inp_path):
            self._load_inp_weights(model, inp_path)
        return model

    def _load_inp_weights(self, model, path):
        """Map INP's single encoder into both MLP_D and MLP_K."""
        src = torch.load(path, map_location="cpu")
        dst = model.state_dict()
        loaded = 0

        for key, val in src.items():
            if key.startswith("latent_encoder.encoder."):
                suffix = key[len("latent_encoder.encoder."):]
                for prefix in ("latent_encoder.data_encoder.",
                               "latent_encoder.knowledge_latent_encoder."):
                    tgt = prefix + suffix
                    if tgt in dst and dst[tgt].shape == val.shape:
                        dst[tgt] = val.clone()
                        loaded += 1
            elif "knowledge_merger" in key:
                continue
            elif key in dst and dst[key].shape == val.shape:
                dst[key] = val
                loaded += 1

        model.load_state_dict(dst)
        print(f"[INP init] {loaded} params from {path}")

    def _after_backward(self):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)


class BINPTwoPhaseTrainer(BINPTrainer):
    """Two-phase training with robustness-aware checkpoint selection.

    Checkpoints during Phase 2:
        model_best.pt   — best validation ELBO
        model_v0.pt     — best NLL at delta=0
        model_v3.pt     — best 0.5*NLL(d=0) + 0.5*NLL(d=0.5)
        model_phase1.pt — snapshot at Phase 1 to Phase 2 transition
    """

    def __init__(self, config, save_dir, load_path=None, last_save_it=0):
        self.phase2_epoch = config.phase2_epoch
        self.phase2_lr_factor = config.phase2_lr_factor
        self.phase2_disagree_lr = getattr(config, "phase2_disagree_lr", None)
        self.p_shuffle = config.p_shuffle
        self.p_absent = config.p_absent
        self._in_phase2 = False
        self._epoch = 0
        self._best = {"v0": np.inf, "v3": np.inf}
        self._perturb_fn = getattr(config, "_perturb_fn", None)
        self._eval_deltas = getattr(config, "eval_deltas", [0.0, 0.1, 0.5, 1.0])
        super().__init__(config, save_dir, load_path, last_save_it)

    def run_batch_train(self, batch):
        (x_c, y_c), (x_t, y_t), knowledge, _ = batch
        x_c, y_c = x_c.to(self.device), y_c.to(self.device)
        x_t, y_t = x_t.to(self.device), y_t.to(self.device)
        if self._in_phase2:
            knowledge = corrupt_knowledge(knowledge, self.p_shuffle, self.p_absent, self.model.training)
        return self.get_loss(x_c, y_c, x_t, y_t, knowledge)

    def eval(self):
        losses, val_loss = super().eval()
        ed = self._eval_deltas
        pfn = self._perturb_fn
        d0 = _eval_nll_at_delta(self.model, self.val_dataloader, ed[0], self.device, perturb_fn=pfn)
        d05 = _eval_nll_at_delta(self.model, self.val_dataloader, ed[2], self.device, perturb_fn=pfn)

        checkpoints = {
            "v0": d0,
            "v3": 0.5 * d0 + 0.5 * d05,
        }
        for name, score in checkpoints.items():
            if score < self._best[name]:
                self._best[name] = score
                torch.save(self.model.state_dict(), f"{self.save_dir}/model_{name}.pt")

        return losses, val_loss

    def _start_phase2(self):
        le = self.model.latent_encoder
        disagree_lr = self.phase2_disagree_lr or self.config.lr
        base_lr = self.config.lr * self.phase2_lr_factor

        disagree = {id(le.disagree_scale), id(le.disagree_bias)}
        mlp_k = {id(p) for p in le.knowledge_latent_encoder.parameters()}

        self.optimizer = torch.optim.Adam([
            {"params": [le.disagree_scale, le.disagree_bias], "lr": disagree_lr},
            {"params": list(le.knowledge_latent_encoder.parameters()), "lr": base_lr},
            {"params": [p for p in self.model.parameters()
                        if p.requires_grad and id(p) not in disagree and id(p) not in mlp_k],
             "lr": base_lr},
        ])
        self._in_phase2 = True
        print(f"[Phase 2] disagree_lr={disagree_lr}, base_lr={base_lr}")

    def train(self):
        it = 0
        best_elbo = np.inf
        patience = getattr(self.config, "early_stopping_patience", 0)
        stale = 0

        for epoch in range(self.num_epochs + 1):
            self._epoch = epoch

            if epoch == self.phase2_epoch and not self._in_phase2:
                torch.save(self.model.state_dict(), f"{self.save_dir}/model_phase1.pt")
                self._start_phase2()
                best_elbo = np.inf
                stale = 0
                self._best = {"v0": np.inf, "v3": np.inf}

            stop = False
            for batch in self.train_dataloader:
                self.model.train()
                self.optimizer.zero_grad()
                loss = self.run_batch_train(batch)["loss"]
                loss.backward()
                self._after_backward()
                self.optimizer.step()

                if it % EVAL_ITER == 0 and it > 0:
                    _, val = self.eval()
                    if val < best_elbo:
                        best_elbo = val
                        stale = 0
                        torch.save(self.model.state_dict(), f"{self.save_dir}/model_best.pt")
                        torch.save(self.optimizer.state_dict(), f"{self.save_dir}/optim_best.pt")
                        print(f"Best model saved at iteration {self.last_save_it + it}")
                    else:
                        stale += 1
                    if self._in_phase2 and patience > 0 and stale >= patience:
                        print(f"Early stopping at iteration {it}")
                        stop = True
                        break
                it += 1
            if stop:
                break

        return best_elbo


class INPCTrainer(Trainer):
    """Standard INP trained with corruption. Optional INP checkpoint init."""

    def __init__(self, config, save_dir, load_path=None, last_save_it=0):
        self.p_shuffle = config.p_shuffle
        self.p_absent = config.p_absent
        super().__init__(config, save_dir, load_path, last_save_it)

    def _create_model(self, config):
        model = super()._create_model(config)
        inp_path = getattr(config, "inp_init_path", None)
        if inp_path and os.path.exists(inp_path):
            model.load_state_dict(torch.load(inp_path, map_location="cpu"))
            print(f"[INP init] Loaded from {inp_path}")
        return model

    def run_batch_train(self, batch):
        (x_c, y_c), (x_t, y_t) = batch[0], batch[1]
        x_c, y_c = x_c.to(self.device), y_c.to(self.device)
        x_t, y_t = x_t.to(self.device), y_t.to(self.device)
        knowledge = corrupt_knowledge(batch[2], self.p_shuffle, self.p_absent, self.model.training)
        return self.get_loss(x_c, y_c, x_t, y_t, knowledge)


def _next_run_number(save_dir, prefix):
    if not os.path.exists(save_dir):
        return 0
    pat = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    nums = [int(m.group(1)) for d in os.listdir(save_dir) if (m := pat.match(d))]
    return max(nums) + 1 if nums else 0


def meta_train_binp(trial, config, run_name_prefix="run"):
    config.device = get_device()
    root = f"./saves/{config.project_name}"
    os.makedirs(root, exist_ok=True)
    n = _next_run_number(root, run_name_prefix)
    save_dir = f"{root}/{run_name_prefix}_{n}"
    os.makedirs(save_dir, exist_ok=True)

    trainers = {"binp": BINPTrainer, "binp_twophase": BINPTwoPhaseTrainer, "inp_c": INPCTrainer}
    cls = trainers.get(getattr(config, "model_type", "inp"), Trainer)
    trainer = cls(config=config, save_dir=save_dir)
    _non_serial = {k: config.__dict__.pop(k) for k in list(config.__dict__)
                   if callable(config.__dict__[k]) and not isinstance(config.__dict__[k], type)}
    trainer.config.write_config(f"{save_dir}/config.toml")
    config.__dict__.update(_non_serial)
    return trainer.train()
