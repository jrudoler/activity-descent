"""Small, independent NumPy reference used only for validation."""

import numpy as np


def forward(weights, decoder, x):
    activities, preactivations = [x], []
    for w in weights:
        z = w @ np.append(activities[-1], 1.)
        preactivations.append(z)
        activities.append(np.maximum(z, 0))
    return decoder @ activities[-1], activities, preactivations


def derivatives(weights, decoder, x, error):
    _, activities, zs = forward(weights, decoder, x)
    g = decoder.T @ error
    gradients, weight_gradients = [None] * len(weights), [None] * len(weights)
    for layer in range(len(weights) - 1, -1, -1):
        gradients[layer] = g.copy()
        local = g * (zs[layer] > 0)
        weight_gradients[layer] = np.outer(local, np.append(activities[layer], 1.))
        if layer:
            g = weights[layer][:, :-1].T @ local
    return gradients, weight_gradients


def activity_jacobians(weights, decoder, x):
    """Construct actual parameter Jacobians, independently of kernel recursion."""
    _, activities, zs = forward(weights, decoder, x)
    total = sum(w.size for w in weights)
    previous, offset, jacobians = None, 0, []
    for layer, w in enumerate(weights):
        derivative = (zs[layer] > 0).astype(float)
        jacobian = np.zeros((w.shape[0], total))
        augmented = np.append(activities[layer], 1.)
        if previous is not None:
            jacobian += (derivative[:, None] * w[:, :-1]) @ previous
        for row in range(w.shape[0]):
            start = offset + row * w.shape[1]
            jacobian[row, start:start + w.shape[1]] = derivative[row] * augmented
        offset += w.size
        jacobians.append(jacobian)
        previous = jacobian
    return jacobians
