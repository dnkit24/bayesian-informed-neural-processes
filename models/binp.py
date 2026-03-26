"""Bayesian Informed Neural Process (BINP).

    Data:        MLP_D(ReLU(R))  -> (mu_D, sigma_D)
    Knowledge:   MLP_K(ReLU(k))  -> (mu_K, sigma_K_base)
    Reliability: sigma_K = sigma_K_base + softplus(softplus(scale) * |mu_D - mu_K| + bias)
    Combination: Gaussian product(q_D, q_K) -> (mu_*, sigma_*)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))
from models.modules import MLP, KnowledgeEncoder
from models.inp import INP
from models.utils import MultivariateNormalDiag


class BINPLatentEncoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        h = config.hidden_dim
        n_hidden = config.latent_encoder_num_hidden
        self.hidden_dim = h

        self.data_encoder = MLP(h, h, n_hidden, 2 * h)

        if config.use_knowledge:
            self.knowledge_encoder = KnowledgeEncoder(config)
            self.knowledge_latent_encoder = MLP(h, h, n_hidden, 2 * h)
            self.disagree_scale = nn.Parameter(torch.full((h,), getattr(config, 'disagree_scale_init', 0.54)))
            self.disagree_bias = nn.Parameter(torch.full((h,), config.disagree_bias_init))
        else:
            self.knowledge_encoder = None

        self.knowledge_dropout = getattr(config, 'knowledge_dropout', 0)
        self._diag = {}

    def forward(self, R, knowledge, n):
        mu_D, sigma_D = self._encode(self.data_encoder, R)

        if self.training and self.knowledge_dropout > 0 and torch.rand(1).item() < self.knowledge_dropout:
            knowledge = None

        if knowledge is None or self.knowledge_encoder is None:
            self._diag = self._make_diag(mu_D, sigma_D)
            return mu_D, sigma_D

        k = self.knowledge_encoder(knowledge)
        mu_K, sigma_K_base = self._encode(self.knowledge_latent_encoder, k)

        disagreement = (mu_D.detach() - mu_K.detach()).abs()
        scale = F.softplus(self.disagree_scale)
        sigma_K = sigma_K_base + F.softplus(scale * disagreement + self.disagree_bias)

        log_prec_D = -2.0 * torch.log(sigma_D)
        log_prec_K = -2.0 * torch.log(sigma_K)
        log_prec = torch.logaddexp(log_prec_D, log_prec_K)
        sigma_star = torch.exp(-0.5 * log_prec)
        w_D = torch.exp(log_prec_D - log_prec)
        w_K = torch.exp(log_prec_K - log_prec)
        mu_star = w_D * mu_D + w_K * mu_K

        self._diag = self._make_diag(mu_D, sigma_D, mu_K, sigma_K)
        return mu_star, sigma_star

    def _encode(self, encoder, x):
        raw = encoder(F.relu(x))
        mu, raw_sigma = raw.split(self.hidden_dim, dim=-1)
        sigma = 0.01 + 0.99 * F.softplus(raw_sigma)
        return mu, sigma

    def _make_diag(self, mu_D, sigma_D, mu_K=None, sigma_K=None):
        d = {"mu_D": mu_D.detach(), "sigma_D": sigma_D.detach()}
        if mu_K is not None:
            d["mu_K"] = mu_K.detach()
            d["sigma_K"] = sigma_K.detach()
            d["variance_ratio"] = (sigma_K ** 2 / sigma_D ** 2).detach()
        return d

    def get_knowledge_embedding(self, knowledge):
        if self.knowledge_encoder is None:
            return None
        k = self.knowledge_encoder(knowledge)
        return k.unsqueeze(1) if k.dim() == 2 else k


class BINP(INP):
    """INP with precision-weighted Bayesian knowledge fusion."""

    def _create_latent_encoder(self, config):
        return BINPLatentEncoder(config)

    def infer_latent_dist(self, R, knowledge, n):
        mu, sigma = self.latent_encoder(R, knowledge, n)
        return MultivariateNormalDiag(mu, sigma)

    def get_knowledge_embedding(self, knowledge):
        return self.latent_encoder.get_knowledge_embedding(knowledge)

    def get_diagnostics(self):
        return dict(self.latent_encoder._diag)
