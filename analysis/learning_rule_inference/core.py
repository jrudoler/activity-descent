"""Batched ReLU MLPs with a fixed, known linear readout.

Leading dimension indexes independent networks. The remaining leading dimensions
index examples, restored-state probes, or perturbation replicates. All updates
are functional: a probe never changes the checkpoint or decoder.
"""

from dataclasses import dataclass
import math

import numpy as np
import torch


@dataclass
class State:
    weights: list[torch.Tensor]
    decoder: torch.Tensor
    feedback: list[torch.Tensor]
    readout_feedback: torch.Tensor
    preconditioner: list[torch.Tensor]
    seeds: list[int]

    def to(self, device, dtype=torch.float64):
        convert = lambda x: x.to(device=device, dtype=dtype)
        return State(
            [convert(w) for w in self.weights], convert(self.decoder),
            [convert(b) for b in self.feedback], convert(self.readout_feedback),
            [convert(p) for p in self.preconditioner], self.seeds.copy(),
        )


def initialize(seeds, width=32, depth=2, outputs=8, inputs=16):
    """Generate identical float64 checkpoints for either computing backend."""
    if not 1 <= outputs <= width or depth < 1:
        raise ValueError("Require 1 <= outputs <= width and depth >= 1")
    sizes = [inputs] + [width] * depth
    networks, decoders, feedback, readout_feedback, preconditioners = [], [], [], [], []
    for seed in seeds:
        rng = np.random.default_rng([int(seed), 0])
        ws, bs, ps = [], [], []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            w = rng.normal(size=(n_out, n_in + 1)) * math.sqrt(2 / n_in)
            w[:, -1] = 0
            ws.append(w)
            bs.append(rng.normal(size=(n_in, n_out)) / math.sqrt(n_out))
            ps.append(np.exp(rng.uniform(-math.log(3), math.log(3), w.shape)))
        q, _ = np.linalg.qr(rng.normal(size=(width, outputs)))
        networks.append(ws)
        decoders.append(q.T)
        feedback.append(bs)
        preconditioners.append(ps)
        readout_feedback.append(rng.normal(size=(width, outputs)) / math.sqrt(outputs))
    stack = lambda xs: torch.from_numpy(np.stack(xs))
    return State(
        [stack([w[l] for w in networks]) for l in range(depth)], stack(decoders),
        [stack([b[l] for b in feedback]) for l in range(depth)],
        stack(readout_feedback),
        [stack([p[l] for p in preconditioners]) for l in range(depth)], list(seeds),
    )


def bar_data(seeds, outputs=8, n_per_class=20, original_targets=False):
    """Original noisy bar inputs, with either class or fixed-teacher targets."""
    all_x, all_y = [], []
    for seed in seeds:
        rng = np.random.default_rng([int(seed), 1])
        xs, labels = [], []
        for _ in range(n_per_class):
            horizontal = np.zeros((4, 4))
            horizontal[rng.integers(4), :] = 1
            xs.append(horizontal.ravel() + rng.normal(0, .15, 16))
            labels.append([1., 0.])
            vertical = np.zeros((4, 4))
            vertical[:, rng.integers(4)] = 1
            xs.append(vertical.ravel() + rng.normal(0, .15, 16))
            labels.append([0., 1.])
        x = np.asarray(xs)
        teacher = np.random.default_rng([int(seed), 2]).normal(size=(outputs, 16)) / 4
        if original_targets and outputs != 2:
            raise ValueError("Class targets require two outputs")
        all_x.append(x)
        all_y.append(np.asarray(labels) if original_targets else x @ teacher.T)
    return torch.from_numpy(np.stack(all_x)), torch.from_numpy(np.stack(all_y))


def matvec(matrix, vector):
    """Broadcast network matrices over additional sample dimensions."""
    while matrix.ndim < vector.ndim + 1:
        matrix = matrix.unsqueeze(-3)
    return (matrix @ vector.unsqueeze(-1)).squeeze(-1)


def forward(state, x, weights=None, noise=None):
    weights = state.weights if weights is None else weights
    activities, preactivations = [x], []
    for layer, w in enumerate(weights):
        z = matvec(w[..., :-1], activities[-1])
        bias = w[..., -1]
        while bias.ndim < z.ndim:
            bias = bias.unsqueeze(-2)
        z = z + bias
        if noise is not None and noise[layer] is not None:
            z = z + noise[layer]
        preactivations.append(z)
        activities.append(z.clamp_min(0))
    return matvec(state.decoder, activities[-1]), activities, preactivations


def activity_gradients(state, errors, preactivations):
    g = matvec(state.decoder.transpose(-1, -2), errors)
    gradients = [None] * len(state.weights)
    for layer in reversed(range(len(state.weights))):
        gradients[layer] = g
        if layer:
            g = matvec(
                state.weights[layer][..., :-1].transpose(-1, -2),
                g * (preactivations[layer] > 0),
            )
    return gradients


def directions(state, x, errors, rule="gd", repeats=1, sigma=1e-3,
               generator=None, shared_noise=False):
    """Return weight *descent directions* before multiplication by -eta.

    Node perturbation is a central-difference layerwise isotropic directional
    derivative estimator: n (L(+eps)-L(-eps)) eps / (2 ||eps||^2), averaged
    over repeats, times the clean presynaptic activity. Each layer is perturbed
    separately from the same clean state. Its small-sigma expectation is the
    gradient. shared_noise uses the same perturbations across the probe bank.
    """
    output, acts, zs = forward(state, x)
    if rule == "node":
        if repeats < 1 or sigma <= 0 or x.ndim != 3:
            raise ValueError("Node probes require [network, probe, input], R>=1, sigma>0")
        target = output - errors
        xr = x.unsqueeze(-2).expand(*x.shape[:-1], repeats, x.shape[-1])
        updates = []
        for layer, z in enumerate(zs):
            n = z.shape[-1]
            shape = (x.shape[0], 1 if shared_noise else x.shape[1], repeats, n)
            eps = sigma * torch.randn(shape, device=x.device, dtype=x.dtype, generator=generator)
            noises = [None] * len(zs)
            noises[layer] = eps
            plus = forward(state, xr, noise=noises)[0]
            noises[layer] = -eps
            minus = forward(state, xr, noise=noises)[0]
            target_r = target.unsqueeze(-2)
            difference = .5 * ((plus - target_r).square().sum(-1)
                               - (minus - target_r).square().sum(-1))
            signal = (n * difference.unsqueeze(-1) * eps
                      / (2 * eps.square().sum(-1, keepdim=True))).mean(-2)
            augmented = torch.cat((acts[layer], torch.ones_like(acts[layer][..., :1])), -1)
            updates.append(signal.unsqueeze(-1) * augmented.unsqueeze(-2))
        return updates
    if rule not in {"gd", "preconditioned", "feedback"}:
        raise ValueError(f"Unknown rule: {rule}")
    matrix = state.readout_feedback if rule == "feedback" else state.decoder.transpose(-1, -2)
    g = matvec(matrix, errors)
    updates = [None] * len(state.weights)
    for layer in reversed(range(len(state.weights))):
        local = g * (zs[layer] > 0)
        augmented = torch.cat((acts[layer], torch.ones_like(acts[layer][..., :1])), -1)
        update = local.unsqueeze(-1) * augmented.unsqueeze(-2)
        if rule == "preconditioned":
            p = state.preconditioner[layer]
            while p.ndim < update.ndim:
                p = p.unsqueeze(-3)
            update = update * p
        updates[layer] = update
        if layer:
            matrix = (state.feedback[layer] if rule == "feedback"
                      else state.weights[layer][..., :-1].transpose(-1, -2))
            g = matvec(matrix, local)
    return updates


def apply_step(state, updates, eta, layers=None):
    """Return new weights for every probe, leaving the source state untouched."""
    result = []
    for layer, (w, u) in enumerate(zip(state.weights, updates)):
        while w.ndim < u.ndim:
            w = w.unsqueeze(-3)
        result.append(w - eta * u if layers is None or layer in layers else w)
    return result


def activity_jvp(state, x, updates, layers=None):
    """First-order activity responses J dW, with optional parameter freezing."""
    _, acts, zs = forward(state, x)
    previous = torch.zeros_like(x)
    result = []
    for layer, (w, update) in enumerate(zip(state.weights, updates)):
        propagated = matvec(w[..., :-1], previous)
        if layers is None or layer in layers:
            direct = matvec(update[..., :-1], acts[layer]) + update[..., -1]
            propagated = propagated + direct
        previous = propagated * (zs[layer] > 0)
        result.append(previous)
    return result


def kernels(state, x, preconditioned=False):
    """Exact first-order kernels for each hidden layer, with decoder fixed."""
    _, acts, zs = forward(state, x)
    previous, result = None, []
    for layer, w in enumerate(state.weights):
        gate = (zs[layer] > 0).to(x.dtype)
        augmented = torch.cat((acts[layer], torch.ones_like(acts[layer][..., :1])), -1)
        if preconditioned:
            p = state.preconditioner[layer]
            while p.ndim < augmented.ndim + 1:
                p = p.unsqueeze(-3)
            diagonal = (p * augmented.square().unsqueeze(-2)).sum(-1) * gate
        else:
            diagonal = augmented.square().sum(-1, keepdim=True) * gate
        current = torch.diag_embed(diagonal)
        if previous is not None:
            matrix = w[..., :-1]
            while matrix.ndim < previous.ndim:
                matrix = matrix.unsqueeze(-3)
            b = gate.unsqueeze(-1) * matrix
            current = current + b @ previous @ b.transpose(-1, -2)
        previous = current
        result.append(current)
    return result


def probe_errors(seeds, outputs, heldout=16):
    """A spanning +/- basis for fitting and independent unit mixtures for testing."""
    banks = []
    for seed in seeds:
        rng = np.random.default_rng([int(seed), 3])
        mixtures = rng.normal(size=(heldout, outputs))
        mixtures /= np.linalg.norm(mixtures, axis=-1, keepdims=True)
        banks.append(np.concatenate((np.eye(outputs), -np.eye(outputs), mixtures)))
    return torch.from_numpy(np.stack(banks)), 2 * outputs


def train(state, x, target, steps, eta, rule="gd", repeats=1, sigma=1e-3, seed=0):
    """Online learning, batched across independent networks with a frozen decoder."""
    generator = torch.Generator(device=x.device).manual_seed(seed)
    current = state
    history = []
    for step in range(steps):
        indices = torch.randint(x.shape[1], (x.shape[0],), device=x.device, generator=generator)
        rows = torch.arange(x.shape[0], device=x.device)
        sampled_x, sampled_y = x[rows, indices].unsqueeze(1), target[rows, indices].unsqueeze(1)
        errors = forward(current, sampled_x)[0] - sampled_y
        updates = directions(current, sampled_x, errors, rule, repeats, sigma, generator)
        weights = [w.squeeze(1) for w in apply_step(current, updates, eta)]
        current = State(weights, current.decoder, current.feedback,
                        current.readout_feedback, current.preconditioner, current.seeds)
        if step == 0 or (step + 1) % max(1, steps // 10) == 0:
            loss = .5 * (forward(current, x)[0] - target).square().sum(-1).mean(-1)
            history.append((step + 1, loss.detach().cpu().tolist()))
    return current, history
