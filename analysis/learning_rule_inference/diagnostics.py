"""Layerwise activity diagnostics with explicit undefined and tiny-value handling."""

import torch

from . import core


def cosine(a, b):
    denominator = torch.linalg.vector_norm(a, dim=-1) * torch.linalg.vector_norm(b, dim=-1)
    return torch.where(denominator > 1e-20, (a * b).sum(-1) / denominator, torch.nan)


def alignment(g, v, atol=1e-10, rtol=1e-8):
    """v=-Delta A/eta; positive g_i v_i means downhill activity movement."""
    g_tiny = g.abs() <= atol + rtol * g.abs().amax(-1, keepdim=True)
    v_tiny = v.abs() <= atol + rtol * v.abs().amax(-1, keepdim=True)
    informative = ~(g_tiny | v_tiny)
    count = informative.sum(-1)
    product = g * v
    downhill = ((product > 0) & informative).sum(-1)
    fraction = torch.where(count > 0, downhill / count.clamp_min(1), torch.nan)
    return {
        "cosine": cosine(g, v),
        "pearson": cosine(g - g.mean(-1, keepdim=True), v - v.mean(-1, keepdim=True)),
        "downhill_fraction": fraction,
        "uphill_fraction": torch.where(count > 0, 1 - fraction, torch.nan),
        "informative_neurons": count,
        "tiny_gradient_neurons": g_tiny.sum(-1),
        "tiny_response_neurons": v_tiny.sum(-1),
        "population_descent_rate": product.sum(-1),
    }


def step_diagnostics(state, x, errors, eta, layers=None):
    """Known-GD mechanism measurements; inference does not consume these kernels."""
    output_before, before, zs = core.forward(state, x)
    updates = core.directions(state, x, errors)
    changed = core.apply_step(state, updates, eta, layers)
    output_after, after, zs_after = core.forward(state, x, changed)
    gradients = core.activity_gradients(state, errors, zs)
    predictions = core.activity_jvp(state, x, updates, layers)
    kernels = core.kernels(state, x, layers=layers)
    decoder = state.decoder.unsqueeze(1)
    kernels.append(decoder @ kernels[-1] @ decoder.transpose(-1, -2))
    gradients.append(errors)
    predictions.append(core.matvec(state.decoder, predictions[-1]))
    before = before[1:] + [output_before]
    after = after[1:] + [output_after]
    result = []
    for layer, (g, k, prediction) in enumerate(zip(gradients, kernels, predictions)):
        v = -(after[layer] - before[layer]) / eta
        norm = torch.linalg.vector_norm(v, dim=-1)
        self_term = k.diagonal(dim1=-2, dim2=-1) * g
        h = core.matvec(k, g) - self_term
        is_output = layer == len(state.weights)
        active = torch.ones_like(g, dtype=torch.bool) if is_output else zs[layer] > 0
        gate_on = torch.zeros_like(g, dtype=torch.bool) if is_output else (zs[layer] <= 0) & (zs_after[layer] > 0)
        gate_off = torch.zeros_like(g, dtype=torch.bool) if is_output else (zs[layer] > 0) & (zs_after[layer] <= 0)
        self_abs, h_abs = self_term.abs(), h.abs()
        valid_self = self_abs > 1e-10 + 1e-8 * self_abs.amax(-1, keepdim=True)
        ratio = torch.where(valid_self, h_abs / self_abs.clamp_min(1e-30), torch.nan)
        result.append({
            "layer": layer,
            "layer_type": "output" if is_output else "hidden",
            **alignment(g, v),
            "active_neurons": active.sum(-1),
            "inactive_neurons": (~active).sum(-1),
            "gate_on": gate_on.sum(-1),
            "gate_off": gate_off.sum(-1),
            "full_prediction_cosine": cosine(prediction, v),
            "diagonal_prediction_cosine": cosine(self_term, v),
            "full_relative_error": torch.where(norm > 1e-20,
                torch.linalg.vector_norm(v - prediction, dim=-1) / norm.clamp_min(1e-30), torch.nan),
            "diagonal_relative_error": torch.where(norm > 1e-20,
                torch.linalg.vector_norm(v - self_term, dim=-1) / norm.clamp_min(1e-30), torch.nan),
            "gradient_dot_activity_change": -eta * (g * v).sum(-1),
            "interference_l2": torch.linalg.vector_norm(h, dim=-1),
            "self_term_l2": torch.linalg.vector_norm(self_term, dim=-1),
            "median_interference_ratio": ratio.nanmedian(-1).values,
            "self_dominates_fraction": torch.where(valid_self.sum(-1) > 0,
                ((h_abs < self_abs) & valid_self).sum(-1) / valid_self.sum(-1).clamp_min(1), torch.nan),
            "self_ratio_valid_neurons": valid_self.sum(-1),
            "updated_parameter_layers": list(range(len(state.weights))) if layers is None else layers,
        })
    return result
