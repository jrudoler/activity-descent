# Learning-rule inference: staged results

Updated 2026-10-21. This is a local research note, not approved main-paper text.
The original manuscript remains unchanged at its handoff version.

## Question and observation model

Can controlled activity responses distinguish update-rule classes when the
upstream weights are unknown? At a restored state, first-order GD predicts
`v = K g`, with one shared symmetric PSD `K`. The experiment observes a complete
hidden layer, knows the fixed decoder and loss, and varies the target. It does
not supply the true kernel to inference. These are restrictive assumptions.

## Main result

The locked CPU study used 200 calibration and 200 evaluation networks per
profile, across eight profiles. At the primary width-64, depth-two,
eight-output, 1000-step checkpoint, without added observation noise:

| Quantity | CPU evaluation | CUDA replication |
| --- | ---: | ---: |
| Operator detection of feedback mixture 0.1 | 200/200 | 200/200 |
| Pearson detection of that rule | 6/200 | 11/200 |
| Operator rejection of ordinary GD | 11/200 | 12/200 |
| Operator rejection of the fixed PSD preconditioner | 11/200 | 10/200 |

The CPU paired operator-minus-Pearson gain is 97.0 percentage points, with the
prespecified approximate conservative interval [90.5, 98.8]. The CUDA replication
uses the original CPU cutoffs. Its initialization/data identities overlap the
CPU evaluation, but training RNG streams differ; do not pool these as 400
independent networks. See the exact CSVs for per-condition Wilson intervals.

## What the result does and does not identify

- A shared PSD constraint can expose feedback bias missed by scalar alignment.
  It does not uniquely identify GD: fixed PSD preconditioning is an equivalence.
- Power is conditional on state and observation quality. CPU depth-three power
  is 30/200; adding 1% GD-RMS noise gives 69/200, versus 74/200 in CUDA.
- Calibration on GD and one preconditioner does not cover every PSD update.
  Shared one-sample node rejection under noise is 176/200 on CPU and 177/200
  on CUDA. This limitation is part of the result.
- A stochastic bank can fail a common-operator test while its conditional mean
  approaches GD. The separate uncertainty-aware mean analysis accepts most
  unbiased controls at large repeat counts; it tests a necessary reciprocity
  condition, not a complete PSD model or unbiasedness certificate.
- Singular PSD fitting can have an unattained infimum. Keep attainment flags,
  undefined predictions, and the supported gradient span explicit.

## Numerical and compute checks

All 18 scientific checks pass on CPU, a B200 MIG 2g.45gb slice, and a full B200.
At identical saved CPU checkpoints, the CUDA comparison has maximum relative
response error 1.65e-12 and zero changed operator decisions across four
deterministic rules, three checkpoints, and 200 networks.

Full-device job 9156748 completed on October 20, 14:40:53--14:42:07 Eastern.
It ran for 74 seconds after waiting 23h 42m 38s from submission. Matched
single-core CPU and actual GPU reports retain source hashes, runtime, precision,
device, and timing provenance. Small operator calls favor CPU; batched update
simulation favors GPU. The reports are observations from individual allocations,
not typical queue-delay or end-to-end training estimates.

## Evidence and possible promotion

Primary evidence is `results/data/inference_evaluation.csv` and its provenance;
CUDA evidence is `inference_gpu_evaluation.csv` and
`inference_backend_validation.csv`. Hardware evidence is in the matching
benchmark CSVs and archived original reports. [The research index](RESEARCH_INDEX.md)
links the protocols, theory, controls, and detailed execution record.

All material remains staged. The user will decide which finding, explanation,
figure, or appendix should be promoted into the main manuscript.
