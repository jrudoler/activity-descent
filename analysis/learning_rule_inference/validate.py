"""Scientific implementation checks; run on CPU and again on an allocated GPU."""

import argparse
from pathlib import Path
import time
import unittest

import numpy as np
import torch

from . import campaign, core, diagnostics, evaluate, operators, reference
from .provenance import provenance, save_json

DEVICE = "cpu"


class ScientificChecks(unittest.TestCase):
    def state(self, width=8, depth=2, outputs=2):
        return core.initialize([10, 11], width, depth, outputs).to(DEVICE)

    def assert_close(self, x, y, atol=1e-9, rtol=1e-9):
        torch.testing.assert_close(torch.as_tensor(x).cpu(), torch.as_tensor(y).cpu(), atol=atol, rtol=rtol)

    def test_numpy_forward_and_backprop(self):
        state = self.state()
        x = core.bar_data(state.seeds, outputs=2)[0][:, :2].to(DEVICE)
        errors = torch.tensor([[[1., -.4], [.3, .8]]] * 2, device=DEVICE, dtype=torch.float64)
        output, acts, zs = core.forward(state, x)
        updates = core.directions(state, x, errors)
        gradients = core.activity_gradients(state, errors, zs)
        for b in range(2):
            weights = [w[b].cpu().numpy() for w in state.weights]
            decoder = state.decoder[b].cpu().numpy()
            for p in range(2):
                xn, en = x[b, p].cpu().numpy(), errors[b, p].cpu().numpy()
                yn, an, zn = reference.forward(weights, decoder, xn)
                gn, wn = reference.derivatives(weights, decoder, xn, en)
                self.assert_close(output[b, p], yn)
                for l in range(2):
                    self.assert_close(acts[l + 1][b, p], an[l + 1])
                    self.assert_close(zs[l][b, p], zn[l])
                    self.assert_close(updates[l][b, p], wn[l])
                    self.assert_close(gradients[l][b, p], gn[l])

    def test_autograd_and_finite_difference(self):
        state = self.state()
        x = core.bar_data(state.seeds, outputs=2)[0][:, :1].to(DEVICE)
        weights = [w.clone().requires_grad_() for w in state.weights]
        output = core.forward(state, x, weights)[0]
        target = torch.full_like(output, .3)
        loss = .5 * (output - target).square().sum()
        expected = torch.autograd.grad(loss, weights)
        actual = core.directions(state, x, output.detach() - target)
        for l in range(2):
            self.assert_close(actual[l].squeeze(1), expected[l])
        eps = 1e-6
        perturb = [torch.ones_like(w) * .03 for w in weights]
        def objective(sign):
            y = core.forward(state, x, [w.detach() + sign * eps * p for w, p in zip(weights, perturb)])[0]
            return .5 * (y - target).square().sum()
        numerical = (objective(1) - objective(-1)) / (2 * eps)
        exact = sum((g * p).sum() for g, p in zip(expected, perturb))
        self.assert_close(numerical, exact, atol=2e-8, rtol=2e-8)

    def test_numpy_jacobian_kernels(self):
        state = self.state()
        x = core.bar_data(state.seeds, outputs=2)[0][:, :1].to(DEVICE)
        for preconditioned in (False, True):
            kernels = core.kernels(state, x, preconditioned)
            for b in range(2):
                js = reference.activity_jacobians(
                    [w[b].cpu().numpy() for w in state.weights],
                    state.decoder[b].cpu().numpy(), x[b, 0].cpu().numpy(),
                )
                diagonal = np.concatenate([p[b].cpu().numpy().ravel() for p in state.preconditioner])
                for l, j in enumerate(js):
                    expected = (j * diagonal) @ j.T if preconditioned else j @ j.T
                    self.assert_close(kernels[l][b, 0], expected)

    def test_restoration_and_taylor_convergence(self):
        state = self.state()
        original = [w.clone() for w in state.weights]
        x = core.bar_data(state.seeds, outputs=2)[0][:, :1].to(DEVICE)
        errors = torch.ones((2, 1, 2), device=DEVICE, dtype=torch.float64)
        _, acts, zs = core.forward(state, x)
        g = core.activity_gradients(state, errors, zs)[-1]
        for rule in ("gd", "preconditioned"):
            k = core.kernels(state, x, rule == "preconditioned")[-1]
            updates = core.directions(state, x, errors, rule)
            residuals = []
            for eta in (1e-3, 1e-4, 1e-5):
                changed = core.forward(state, x, core.apply_step(state, updates, eta))[1][-1] - acts[-1]
                residuals.append(torch.linalg.vector_norm(-changed / eta - core.matvec(k, g)).item())
            self.assertLess(residuals[-1], residuals[0] * .03)
        for before, after in zip(original, state.weights):
            self.assertTrue(torch.equal(before, after))

    def test_own_upstream_decomposition(self):
        state = self.state()
        x = core.bar_data(state.seeds, outputs=2)[0][:, :1].to(DEVICE)
        errors = torch.ones((2, 1, 2), device=DEVICE, dtype=torch.float64)
        updates = core.directions(state, x, errors)
        own = core.activity_jvp(state, x, updates, [1])[-1]
        upstream = core.activity_jvp(state, x, updates, [0])[-1]
        combined = core.activity_jvp(state, x, updates)[-1]
        self.assert_close(own + upstream, combined)
        k_own = core.kernels(state, x, layers=[1])[-1]
        k_upstream = core.kernels(state, x, layers=[0])[-1]
        self.assert_close(k_own + k_upstream, core.kernels(state, x)[-1])
        self.assert_close(k_own, torch.diag_embed(k_own.diagonal(dim1=-2, dim2=-1)))
        eta = 1e-5
        before = core.forward(state, x)[1][-1]
        after = core.forward(state, x, core.apply_step(state, updates, eta, [1]))[1][-1]
        self.assert_close(-(after - before) / eta, own, atol=1e-9)
        measured = diagnostics.step_diagnostics(state, x, errors, eta, [1])
        self.assertEqual(measured[-1]["layer_type"], "output")
        self.assertLess(measured[-1]["full_relative_error"].max().item(), 1e-8)

    def test_zero_gradient_and_undefined_metrics(self):
        zero = torch.zeros((2, 8), device=DEVICE, dtype=torch.float64)
        result = diagnostics.alignment(zero, zero)
        self.assertTrue(torch.isnan(result["cosine"]).all())
        self.assertTrue(torch.isnan(result["pearson"]).all())
        self.assertTrue(torch.isnan(result["downhill_fraction"]).all())
        self.assertTrue((result["tiny_gradient_neurons"] == 8).all())
        scores = operators.compatibility(zero.unsqueeze(-1), zero.unsqueeze(-1))
        for name in ("reciprocity", "negative_quadratic", "linearity", "nullspace", "gradient_rank"):
            self.assertTrue((scores[name] == 0).all())

    def test_exact_compatibility_including_nullspace(self):
        g = torch.eye(2, device=DEVICE, dtype=torch.float64)
        k = torch.tensor([[2., .4], [.4, 1.]], device=DEVICE, dtype=torch.float64)
        good = operators.compatibility(g, k @ g)
        for name in ("reciprocity", "negative_quadratic", "linearity", "nullspace"):
            self.assertLess(good[name].item(), 1e-10)
        self.assert_close(operators.exact_extension(g, k @ g), k)
        asym = torch.tensor([[1., 1.], [0., 1.]], device=DEVICE, dtype=torch.float64)
        self.assertGreater(operators.compatibility(g, asym)["reciprocity"].item(), .1)
        indefinite = torch.diag(torch.tensor([1., -1.], device=DEVICE, dtype=torch.float64))
        self.assertGreater(operators.compatibility(g, indefinite)["negative_quadratic"].item(), .1)
        # Reciprocity and nonnegative quadratic forms alone miss G=e1,V=e2.
        one_g, one_v = g[:, :1], g[:, 1:]
        incompatible = operators.compatibility(one_g, one_v)
        self.assertEqual(incompatible["reciprocity"].item(), 0.)
        self.assertEqual(incompatible["negative_quadratic"].item(), 0.)
        self.assertAlmostEqual(incompatible["nullspace"].item(), 1.)

    def test_psd_extension_retains_response_outside_probe_span(self):
        g = torch.tensor([[1.], [0.]], device=DEVICE, dtype=torch.float64)
        v = torch.tensor([[1.], [2.]], device=DEVICE, dtype=torch.float64)
        k = operators.exact_extension(g, v)
        self.assert_close(k @ g, v)
        self.assertGreaterEqual(torch.linalg.eigvalsh(k)[0].item(), -1e-10)
        self.assertLess(operators.compatibility(g, v)["nullspace"].item(), 1e-10)

    def test_convex_psd_solver(self):
        g = torch.eye(2, device=DEVICE, dtype=torch.float64)
        incompatible = torch.tensor([[1., 2.], [0., -1.]], device=DEVICE, dtype=torch.float64)
        fitted, diagnostics = operators.fit_psd(g, incompatible, iterations=100)
        self.assert_close(fitted, operators.project_psd(operators.sym(incompatible)))
        self.assertTrue(diagnostics["converged"].item())
        compatible = torch.tensor([[2., .5], [.5, 1.]], device=DEVICE, dtype=torch.float64)
        fitted, diagnostics = operators.fit_psd(g, compatible, iterations=100)
        self.assert_close(fitted, compatible)
        self.assertLess(diagnostics["fit_relative_error"].item(), 1e-10)

    def test_isotropic_span_fit_and_unattained_infimum(self):
        g = torch.tensor([[1., 0., -1., 0.], [0., 1., 0., -1.], [0., 0., 0., 0.]],
                         device=DEVICE, dtype=torch.float64)
        k = torch.tensor([[2., .3, .8], [.3, 1., .4], [.8, .4, 3.]],
                         device=DEVICE, dtype=torch.float64)
        v = k @ g
        restriction, q, result = operators.fit_isotropic_span(g, v)
        self.assertTrue(result["fit_attained"].item())
        self.assert_close(restriction @ (q.T @ g), v)
        self.assertLess(result["fit_relative_error"].item(), 1e-12)
        # A PSD least-squares infimum can be zero without an interpolant.
        one_g = torch.tensor([[1.], [0.]], device=DEVICE, dtype=torch.float64)
        one_v = torch.tensor([[0.], [1.]], device=DEVICE, dtype=torch.float64)
        _, _, result = operators.fit_isotropic_span(one_g, one_v)
        self.assertFalse(result["fit_attained"].item())
        self.assertLess(result["fit_relative_error"].item(), 1e-12)
        predictions = operators.prediction_errors(one_g, one_v, one_g, one_v)
        self.assertTrue(torch.isnan(predictions["psd_prediction_error"]))
        self.assertLess(predictions["psd_infimum_prediction_error"].item(), 1e-12)

    def test_psd_prediction_rejects_unsupported_extrapolation(self):
        g = torch.tensor([[1.], [0.]], device=DEVICE, dtype=torch.float64)
        v = torch.tensor([[1.], [2.]], device=DEVICE, dtype=torch.float64)
        other = torch.tensor([[0.], [1.]], device=DEVICE, dtype=torch.float64)
        predictions = operators.prediction_errors(g, v, other, other)
        self.assertFalse(predictions["prediction_supported"].item())
        self.assertTrue(torch.isnan(predictions["linear_prediction_error"]))
        self.assertTrue(torch.isnan(predictions["psd_prediction_error"]))

    def test_matched_feedback_and_constant_bank_scales(self):
        state = self.state()
        x = core.bar_data(state.seeds, outputs=2)[0][:, :1].to(DEVICE)
        errors, _ = core.probe_errors(state.seeds, 2, heldout=2)
        errors = errors.to(DEVICE)
        x = x.expand(-1, errors.shape[1], -1)
        norm_errors = campaign.magnitude_errors(state.seeds, 2, count=4).to(DEVICE)
        exact_feedback = campaign.align_feedback(state, 0.)
        gd = core.directions(state, x, errors, "gd")
        fb = core.directions(exact_feedback, x, errors, "feedback")
        for expected, actual in zip(gd, fb):
            self.assert_close(expected, actual)
        for rule in ("gd", "preconditioned", "feedback"):
            _, _, controls = campaign.matched_observation(
                campaign.align_feedback(state, .1), x, errors, norm_errors,
                rule, 1e-4, seed=71)
            self.assertEqual(controls["magnitude_scale"].shape, (2,))
            self.assertTrue((controls["magnitude_scale"] > 0).all())
            self.assert_close(controls["magnitude_scale"] * controls["scale_rule_rms"],
                              controls["scale_reference_rms"])
            if rule == "gd":
                self.assert_close(controls["magnitude_scale"], torch.ones(2, dtype=torch.float64))

    def test_calibration_order_and_independent_run_interval(self):
        self.assertEqual(evaluate.conformal_cutoff(list(range(200)), .05), 190.)
        self.assertEqual(evaluate.conformal_cutoff(list(range(19)), .05), 18.)
        self.assertTrue(np.isinf(evaluate.conformal_cutoff(list(range(10)), .05)))
        lower, upper = evaluate.wilson(10, 200)
        self.assertLess(lower, .05)
        self.assertGreater(upper, .05)
        self.assertGreater(lower, .02)
        self.assertLess(upper, .1)

    def test_multiple_directions_reveal_hidden_asymmetry(self):
        g = torch.tensor([[[1., 0., -1., 0.], [0., 1., 0., -1.]]], device=DEVICE, dtype=torch.float64)
        k = torch.tensor([[[1., .2], [0., 1.]]], device=DEVICE, dtype=torch.float64)
        v = k @ g
        result = campaign.score_observation(g.transpose(-1, -2), v.transpose(-1, -2), 4)
        self.assertGreater(result["scores"]["operator"].item(), .01)
        self.assertLess(result["scores"]["single_direction"].item(), 1e-10)

    def test_readout_rank_and_diagonal_dominance_counterexample(self):
        state = self.state(width=8, outputs=2)
        errors, _ = core.probe_errors(state.seeds, 2)
        g = core.matvec(state.decoder.transpose(-1, -2), errors.to(DEVICE)).transpose(-1, -2)
        self.assertTrue(torch.equal(operators.compatibility(g, g)["gradient_rank"],
                                    torch.tensor([2, 2], device=DEVICE)))
        k = torch.tensor([[1., .5], [.5, 1.]], device=DEVICE, dtype=torch.float64)
        gradient = torch.tensor([.1, -1.], device=DEVICE, dtype=torch.float64)
        change = -k @ gradient
        self.assertGreater((gradient * change)[0].item(), 0.)
        self.assertLess((gradient @ change).item(), 0.)

    def test_layerwise_node_estimator_and_shared_noise_control(self):
        state = self.state(width=4, depth=1, outputs=2)
        # Keep a smooth activation region for the estimator check.
        state.weights[0][..., -1] = 4.
        x = torch.ones((2, 1, 16), device=DEVICE, dtype=torch.float64) * .1
        error = torch.tensor([[[.8, -.3]]] * 2, device=DEVICE, dtype=torch.float64)
        generator = torch.Generator(device=DEVICE).manual_seed(43)
        estimate = core.directions(state, x, error, "node", repeats=8192, sigma=1e-4, generator=generator)
        expected = core.directions(state, x, error)
        relative = torch.linalg.vector_norm(estimate[0] - expected[0]) / torch.linalg.vector_norm(expected[0])
        self.assertLess(relative.item(), .06)
        errors, _ = core.probe_errors(state.seeds, 2, heldout=4)
        errors = errors.to(DEVICE)
        xp = x.expand(-1, errors.shape[1], -1)
        _, acts, zs = core.forward(state, xp)
        generator.manual_seed(48)
        updates = core.directions(state, xp, errors, "node", repeats=1, sigma=1e-5,
                                  generator=generator, shared_noise=True)
        eta = 1e-5
        v = -(core.forward(state, xp, core.apply_step(state, updates, eta))[1][-1] - acts[-1]) / eta
        g = core.activity_gradients(state, errors, zs)[-1]
        scores = operators.compatibility(g.transpose(-1, -2), v.transpose(-1, -2), rtol=1e-6)
        self.assertLess(scores["reciprocity"].max().item(), 1e-7)
        self.assertLess(scores["negative_quadratic"].max().item(), 1e-7)


def main():
    global DEVICE
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    DEVICE = args.device
    torch.set_num_threads(args.threads)
    torch.set_float32_matmul_precision("highest")
    metadata = provenance(args.device)
    start = time.perf_counter()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ScientificChecks))
    if args.device == "cuda":
        torch.cuda.synchronize()
    report = {"passed": result.wasSuccessful(), "tests": result.testsRun,
              "failures": [(str(t), detail) for t, detail in result.failures + result.errors],
              "seconds": time.perf_counter() - start, "provenance": metadata}
    if args.output:
        save_json(args.output, report)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
