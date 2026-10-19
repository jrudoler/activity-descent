"""Inference functions take only observed gradients and responses, never J or K."""

import torch


def norm(x):
    return torch.linalg.vector_norm(x, dim=(-2, -1))


def sym(x):
    return (x + x.transpose(-1, -2)) / 2


def project_psd(x):
    values, vectors = torch.linalg.eigh(sym(x))
    return (vectors * values.clamp_min(0).unsqueeze(-2)) @ vectors.transpose(-1, -2)


def compatibility(g, v, rtol=1e-9):
    """Necessary and sufficient exact constraints, measured as residuals.

    Columns are probes. A symmetric PSD K with KG=V exists iff S=G^T V is
    symmetric PSD and ker(S)=ker(V). Sufficiency follows from K=V S^+ V^T.
    Linear consistency is reported separately for interpretation. Rank/nullspace
    diagnostics depend on rtol and need noise calibration for empirical use.
    """
    tiny = torch.finfo(g.dtype).tiny
    s = g.transpose(-1, -2) @ v
    s_sym = sym(s)
    values = torch.linalg.eigvalsh(s_sym)
    scale = (norm(g) * norm(v)).clamp_min(tiny)
    inverse_g = torch.linalg.pinv(g, rtol=rtol)
    inverse_s = torch.linalg.pinv(s_sym, rtol=rtol, hermitian=True)
    identity = torch.eye(s.shape[-1], device=g.device, dtype=g.dtype)
    null_s = identity - inverse_s @ s_sym
    singular = torch.linalg.svdvals(g)
    rank = (singular > rtol * singular[..., :1]).sum(-1)
    return {
        "reciprocity": norm(s - s.transpose(-1, -2)) / (2 * scale),
        "negative_quadratic": (-values[..., 0]).clamp_min(0) / scale,
        "linearity": norm(v - v @ inverse_g @ g) / norm(v).clamp_min(tiny),
        "nullspace": norm(v @ null_s) / norm(v).clamp_min(tiny),
        "gradient_rank": rank,
        "gradient_singular_values": singular,
    }


def exact_extension(g, v, rtol=1e-9):
    """Canonical extension for exactly compatible data; verify constraints first."""
    s = sym(g.transpose(-1, -2) @ v)
    return v @ torch.linalg.pinv(s, rtol=rtol, hermitian=True) @ v.transpose(-1, -2)


def fit_psd(g, v, iterations=1000, tolerance=1e-8, ridge=0.0):
    """Convex projected-gradient fit of .5||KG-V||_F^2 + ridge/2 ||K||_F^2.

    Accelerated projected gradient (FISTA), with a valid Lipschitz step. The
    projected-gradient mapping verifies stationarity at the final feasible K.
    A failed convergence check is never itself a scientific rejection.
    """
    n = g.shape[-2]
    scale = norm(v).clamp_min(torch.finfo(v.dtype).tiny)
    gram = g @ g.transpose(-1, -2)
    vg = v @ g.transpose(-1, -2)
    lipschitz = (torch.linalg.eigvalsh(gram)[..., -1] + ridge).clamp_min(1e-30)
    step = (1 / lipschitz)[..., None, None]
    k = torch.zeros((*g.shape[:-2], n, n), device=g.device, dtype=g.dtype)
    y = k.clone()
    t = 1.
    for _ in range(iterations):
        gradient = sym(y @ gram - vg) + ridge * y
        new = project_psd(y - step * gradient)
        t_new = (1 + (1 + 4 * t * t) ** .5) / 2
        y = new + (t - 1) / t_new * (new - k)
        k, t = new, t_new
    gradient = sym(k @ gram - vg) + ridge * k
    mapping = norm(k - project_psd(k - step * gradient))
    relative_mapping = mapping / norm(k).clamp_min(1.)
    return k, {
        "fit_relative_error": norm(k @ g - v) / scale,
        "projected_gradient": relative_mapping,
        "converged": relative_mapping <= tolerance,
        "iterations": iterations,
        "ridge": ridge,
    }


def fit_isotropic_span(g, v, rtol=1e-9):
    """Semi-analytical PSD Procrustes infimum for an isotropic probe design.

    G G^T = c Q Q^T, where Q spans the gradients. In coordinates [Q,Q_perp],
    write K=[[A,B^T],[B,D]]. Only A and B affect KG. The minimizing restriction
    has A=PSD(sym(Q^T T)) and B=(I-QQ^T)T, T=V G^T Q/c. It has a PSD extension
    iff ker(A) is contained in ker(B); otherwise the infimum is unattained.
    This is the isotropic special case of the PSD Procrustes reduction, not a
    new optimization method. No Jacobian or true kernel enters the calculation.
    """
    u, singular, _ = torch.linalg.svd(g, full_matrices=False)
    ranks = (singular > rtol * singular[..., :1]).sum(-1)
    rank = int(ranks.reshape(-1)[0].item())
    if rank == 0 or not bool((ranks == rank).all()):
        raise ValueError("Isotropic reduction requires a common positive probe rank")
    retained = singular[..., :rank]
    if not bool(((retained / retained[..., :1] - 1).abs() < 1e-7).all()):
        raise ValueError("Probe design is not isotropic on its span")
    q = u[..., :rank]
    coordinates = q.transpose(-1, -2) @ g
    c = retained[..., :1].square().unsqueeze(-1)
    unconstrained = v @ coordinates.transpose(-1, -2) / c
    core = q.transpose(-1, -2) @ unconstrained
    a = project_psd(core)
    b = unconstrained - q @ core
    inverse_a = torch.linalg.pinv(a, rtol=rtol, hermitian=True)
    identity = torch.eye(rank, device=g.device, dtype=g.dtype)
    null_a = identity - inverse_a @ a
    null_component = norm(b @ null_a)
    scale = norm(unconstrained).clamp_min(torch.finfo(g.dtype).tiny)
    attained = null_component <= 1e-7 * scale
    restriction = q @ a + b
    residual = restriction @ coordinates - v
    # The free PSD block can be large; report its size rather than using it to
    # extrapolate or risking cancellation by materializing a full fitted K.
    extension_block = b @ inverse_a @ b.transpose(-1, -2)
    extension_norm = (norm(a).square() + 2 * norm(b).square()
                      + norm(extension_block).square()).sqrt()
    return restriction, q, {
        "fit_method": "isotropic_span_reduction",
        "fit_relative_error": norm(residual) / norm(v).clamp_min(1e-30),
        "fit_attained": attained,
        "extension_null_residual": null_component / scale,
        "extension_frobenius_norm": torch.where(attained, extension_norm, torch.nan),
        "projected_gradient": norm(a - project_psd(a - (a - sym(core)))),
        "converged": torch.ones_like(attained),
        "iterations": 0,
        "ridge": 0.,
        "gradient_rank": ranks,
    }


def prediction_errors(g_fit, v_fit, g_test, v_test, iterations=500):
    unconstrained = v_fit @ torch.linalg.pinv(g_fit, rtol=1e-9)
    support = g_fit @ torch.linalg.pinv(g_fit, rtol=1e-9) @ g_test
    supported = norm(g_test - support) <= 1e-7 * norm(g_test).clamp_min(1e-30)
    try:
        restriction, q, diagnostics = fit_isotropic_span(g_fit, v_fit)
        prediction = restriction @ (q.transpose(-1, -2) @ g_test)
        attained = diagnostics["fit_attained"]
    except ValueError:
        k, diagnostics = fit_psd(g_fit, v_fit, iterations=iterations)
        prediction = k @ g_test
        attained = diagnostics["converged"]
        diagnostics.update({"fit_method": "projected_gradient", "fit_attained": attained})
    diagonal = (g_fit * v_fit).sum(-1) / g_fit.square().sum(-1).clamp_min(1e-30)
    diagonal = diagonal.clamp_min(0)
    denominator = norm(v_test).clamp_min(1e-30)
    return {
        "psd_prediction_error": torch.where(supported & attained,
            norm(prediction - v_test) / denominator, torch.nan),
        "psd_infimum_prediction_error": torch.where(supported,
            norm(prediction - v_test) / denominator, torch.nan),
        "linear_prediction_error": torch.where(supported,
            norm(unconstrained @ g_test - v_test) / denominator, torch.nan),
        "diagonal_prediction_error": norm(diagonal.unsqueeze(-1) * g_test - v_test) / denominator,
        "prediction_supported": supported,
        **diagnostics,
    }
