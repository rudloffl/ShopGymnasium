"""Shared machine policy: one network applied to every machine (MAPPO with parameter sharing).

Per machine: own features + attention over the other machines + shop features -> recurrent cell -> two heads
(product choice incl. idle, priority). A centralised critic pools all machines + shop into one value.
Nothing depends on the number of machines, so the same weights can run any plant size.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class MachinePolicy(nn.Module):
    def __init__(self, machine_dim, shop_dim, n_choices, n_priorities, hidden=128, cell='gru'):
        super().__init__()
        h = hidden
        self.m_enc = nn.Sequential(nn.Linear(machine_dim, h), nn.ReLU())
        self.s_enc = nn.Sequential(nn.Linear(shop_dim, h), nn.ReLU())
        self.attn = nn.MultiheadAttention(h, 4, batch_first=True)
        self.cell_type = cell
        self.rnn = (nn.GRU if cell == 'gru' else nn.LSTM)(3 * h, h)
        self.product_head = nn.Linear(h, n_choices)
        self.priority_head = nn.Linear(h, n_priorities)
        self.critic = nn.Sequential(nn.Linear(2 * h, h), nn.ReLU(), nn.Linear(h, 1))
        self.hidden = h

    def initial_state(self, batch, n_machines, device):
        z = torch.zeros(1, batch * n_machines, self.hidden, device=device)
        return z if self.cell_type == 'gru' else (z, z.clone())

    def forward(self, machines, shop, product_mask, state):
        """machines (T,B,M,md)  shop (T,B,sd)  product_mask (T,B,K+1)  state from initial_state or previous call.
        Returns product logits (T,B,M,K+1), priority logits (T,B,M,P), value (T,B), new state.
        Sequences are whole episodes (all envs reset together), so no mid-sequence state reset is needed."""
        T, B, M, _ = machines.shape
        m = self.m_enc(machines)
        s = self.s_enc(shop)
        mf = m.flatten(0, 1)
        ctx, _ = self.attn(mf, mf, mf, need_weights=False)
        x = torch.cat([m, ctx.view(T, B, M, -1), s.unsqueeze(2).expand(-1, -1, M, -1)], -1)
        z, state = self.rnn(x.reshape(T, B * M, -1), state)
        z = z.view(T, B, M, -1)

        mask = product_mask.unsqueeze(2).expand(-1, -1, M, -1) > 0
        product_logits = self.product_head(z).masked_fill(~mask, -1e9)
        priority_logits = self.priority_head(z)
        value = self.critic(torch.cat([z.mean(2), s], -1)).squeeze(-1)
        return product_logits, priority_logits, value, state


def distributions(product_logits, priority_logits):
    return Categorical(logits=product_logits), Categorical(logits=priority_logits)


def obs_to_tensors(obs, device):
    """Batched gym obs dict (B, ...) -> tensors with a leading time axis of 1."""
    return tuple(torch.as_tensor(obs[k], dtype=torch.float32, device=device).unsqueeze(0)
                 for k in ('machines', 'shop', 'product_mask'))


class TorchPolicy:
    """Wraps a trained MachinePolicy as a (env, obs) -> action callable for D00_Plant.evaluate."""

    def __init__(self, model, deterministic=True, device='cpu'):
        self.model, self.deterministic, self.device = model.eval(), deterministic, device
        self.state = None

    @torch.no_grad()
    def __call__(self, env, obs):
        if env.t == 0:
            self.state = self.model.initial_state(1, env.cfg.n_machines, self.device)
        batched = {k: v[None] for k, v in obs.items()}
        pl, ql, _, self.state = self.model(*obs_to_tensors(batched, self.device), self.state)
        if self.deterministic:
            a_prod, a_prio = pl.argmax(-1), ql.argmax(-1)
        else:
            dp, dq = distributions(pl, ql)
            a_prod, a_prio = dp.sample(), dq.sample()
        return torch.stack([a_prod[0, 0], a_prio[0, 0]], -1).reshape(-1).cpu().numpy()


def load_policy(path, device='cpu'):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = MachinePolicy(**ckpt['model_kwargs']).to(device)
    model.load_state_dict(ckpt['state_dict'])
    return TorchPolicy(model, device=device), ckpt
