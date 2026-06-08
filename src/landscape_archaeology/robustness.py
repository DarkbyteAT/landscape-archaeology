"""Behaviour-anchored robustness probes at a trained point.

A second family of landscape diagnostic, complementary to the spectral
family in ``landscape_archaeology.singular_spectrum``. Both ride the same
mission — measuring properties of the loss landscape at a point — but
they measure different things and use different machinery:

- **Spectral family** (``singular_spectrum``): linearises an operator at
  a point via JVP/VJP power iteration; returns top-k singular values of
  the Jacobian. Operator-agnostic — bind to ``jax.grad(loss_fn)`` for
  curvature, to ``render_fn`` for reparameterisation geometry.

- **Behaviour-anchored family** (``perturbation_curve``,
  ``retraining_noise_floor``): samples Gaussian perturbations of a
  trained parameter pytree and re-evaluates a scalar quantity; returns
  the response curve. Caller-agnostic in the same spirit as the spectral
  family — the library samples noise and orchestrates the sweep but does
  not interpret the parameter pytree, the eval function, or what
  "behaviour" means; that is left to the caller.

Two primitive operations:

- :func:`retraining_noise_floor` repeats a caller-supplied
  ``train_fn(seed) -> Float[Array, ""]`` K times under fresh seeds and
  returns the empirical standard deviation of the resulting scalars. The
  standard naming for this quantity in the loss-landscape literature is
  σ_run: the noise floor against which any perturbation effect must be
  compared to be distinguishable from a retrain.

- :func:`perturbation_curve` sweeps a per-leaf Gaussian perturbation of
  magnitude ``δ × ||leaf||_F`` over a grid of δ values, evaluating a
  caller-supplied ``eval_fn(params) -> Float[Array, ""]`` after each
  perturbation. Returns the matrix of scalar responses, which the caller
  can normalise by σ_run (the canonical behaviour-anchored ratio
  ``Δquantity / σ_run`` from the noise-probe convention).

Both functions are JAX-pure. No assumptions are made about what is being
trained, what the eval scalar represents, or which leaves of the pytree
are perturbed beyond what the caller's ``perturb_filter`` decides.

The Frobenius-relative Gaussian perturbation kind is the one studied in
the noise-probe convention (a leaf with norm ``||w||_F`` is perturbed by
``δ × ||w||_F × N(0, I)``, making δ a dimensionless fraction of the
leaf's own scale). Other perturbation kinds may be added in the future;
they would compose into the same ``perturbation_curve`` shape.
"""

from collections.abc import Callable
from typing import Literal

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PyTree


PerturbationKind = Literal["frob_relative_gaussian"]


def retraining_noise_floor(
    train_fn: Callable[[int], Float[Array, ""]],
    *,
    K: int,
    seed_base: int = 0,
) -> Float[Array, ""]:
    r"""Empirical retraining-noise floor σ_run.

    Runs ``train_fn(seed_base + k)`` for ``k`` in ``range(K)`` and
    returns the sample standard deviation (``ddof=1``) of the K returned
    scalars. The convention follows the noise-probe literature, where
    σ_run sets the floor at which a perturbed network's behaviour is
    statistically indistinguishable from a fresh retrain.

    No assumption is made about what ``train_fn`` does internally — it
    can train a neural net, fit a regression, run a single GD step on a
    closed-form quadratic. The library only requires that it accepts an
    integer seed and returns a scalar.

    Args:
        train_fn: caller-supplied training procedure. Takes one integer
            seed and returns a scalar (the final-state quantity whose
            spread defines the noise floor — typically test loss).
        K: number of independent retrains. K >= 2 for a defined sample
            std; K >= 5 is the recommended minimum for the ratio CI in
            ``perturbation_curve`` consumers to be meaningful.
        seed_base: starting seed. Seeds passed to ``train_fn`` are
            ``seed_base + 0, ..., seed_base + K - 1``.

    Returns:
        Sample standard deviation across the K retrain scalars
        (``ddof=1``).
    """
    if K < 2:
        raise ValueError(f"K must be >= 2 for a defined sample std; got {K}")
    scalars = jnp.stack([jnp.asarray(train_fn(seed_base + k)) for k in range(K)])
    return jnp.std(scalars, ddof=1)


def perturbation_curve(
    params: PyTree,
    eval_fn: Callable[[PyTree], Float[Array, ""]],
    deltas: Float[Array, " n"],
    *,
    K_perturb: int,
    key: Array,
    perturb_filter: Callable[[str], bool] | None = None,
    perturbation_kind: PerturbationKind = "frob_relative_gaussian",
) -> Float[Array, "n K_perturb"]:
    r"""Sweep behaviour-anchored perturbation response over δ × K_perturb.

    For each ``δ`` in ``deltas`` and each ``k`` in ``range(K_perturb)``,
    perturb every selected leaf of ``params`` by
    ``δ × ||leaf||_F × N(0, I)`` (the ``frob_relative_gaussian`` kind)
    using an independent PRNG fold, then call ``eval_fn(perturbed)`` and
    record the scalar.

    Returns the ``(n, K_perturb)`` matrix of scalars. The caller composes
    these with the unperturbed eval and σ_run to derive the canonical
    ratio ``(eval_fn(perturbed) - eval_fn(params)) / sigma_run``.

    ``params`` is treated as a dict-leaved pytree; the library does not
    inspect leaf shapes beyond computing each leaf's Frobenius norm and
    drawing a same-shape Gaussian. The Frobenius-relative perturbation
    kind makes δ dimensionless (fraction of the leaf's own scale), so
    sweep ranges can be specified without knowing absolute magnitudes.

    Args:
        params: the trained parameter pytree to perturb. Must be a
            dict-like leaves structure (the convention in this workspace
            and the noise-probe literature). Internal leaves are
            identified by their dict key for ``perturb_filter``.
        eval_fn: caller-supplied scalar quantity. Takes a perturbed
            pytree of the same shape as ``params`` and returns a scalar
            (typically test loss).
        deltas: 1-D array of perturbation magnitudes, dimensionless
            fractions of leaf Frobenius norms.
        K_perturb: number of independent perturbation seeds per δ. K >= 5
            is the recommended minimum for a meaningful median/IQR.
        key: PRNG key for the sweep. Sub-keys are derived with
            ``jax.random.fold_in`` so the full ``(δ, perturbation_seed,
            leaf_index)`` grid is reproducible from this one key.
        perturb_filter: predicate on leaf name; only leaves where the
            predicate returns ``True`` are perturbed (others stay at the
            trained values). Default: every leaf is perturbed (the
            whole-network sweep in the noise-probe convention). Pass
            ``lambda name: name == leaf_to_probe`` for a per-leaf sweep.
        perturbation_kind: which family of perturbation to apply. Only
            ``"frob_relative_gaussian"`` is implemented in v0.3; the
            argument is present so future kinds (uniform, structured,
            adversarial) can be added without changing the verb.

    Returns:
        ``(len(deltas), K_perturb)`` array of scalars. Row ``i`` is the
        response at ``deltas[i]`` across K_perturb seeds.
    """
    if perturbation_kind != "frob_relative_gaussian":
        raise NotImplementedError(
            f"perturbation_kind={perturbation_kind!r} not implemented; "
            "only 'frob_relative_gaussian' is supported in v0.3"
        )

    selected_leaves: tuple[str, ...] = tuple(n for n in params if perturb_filter is None or perturb_filter(n))
    n_deltas = int(deltas.shape[0])

    def perturb_once(delta: float, sub_key: Array) -> PyTree:
        out = dict(params)
        leaf_keys = jax.random.split(sub_key, len(selected_leaves))
        for k, name in zip(leaf_keys, selected_leaves, strict=True):
            w = params[name]
            frob = jnp.sqrt(jnp.sum(w**2))
            out[name] = w + delta * frob * jax.random.normal(k, w.shape, dtype=w.dtype)
        return out

    rows: list[Float[Array, " K_perturb"]] = []
    for i in range(n_deltas):
        deltai = deltas[i]
        col: list[Float[Array, ""]] = []
        for j in range(K_perturb):
            sub_key = jax.random.fold_in(key, i * K_perturb + j)
            perturbed = perturb_once(float(deltai), sub_key)
            col.append(jnp.asarray(eval_fn(perturbed)))
        rows.append(jnp.stack(col))
    return jnp.stack(rows)


def delta_from_curve(
    deltas: Float[Array, " n"],
    ratios: Float[Array, "n K_perturb"],
    *,
    c: float,
) -> tuple[float | None, str]:
    r"""Derive δ_leaf(c): the largest δ at which median ratio < c.

    Convention follows the noise-probe summary: ratios are the per-(δ,
    seed) values of ``(perturbed_eval - target_eval) / sigma_run``, and
    ``δ_leaf(c)`` is the largest δ on the sweep grid where the median
    across seeds stays below ``c``. The two well-behaved corners (curve
    never crosses, curve always below) are reported as a note string so
    the caller can decide whether to widen the sweep range.

    Args:
        deltas: 1-D sweep grid of perturbation magnitudes.
        ratios: ``(n_deltas, K_perturb)`` response matrix.
        c: the multiplicative tolerance in units of σ_run.

    Returns:
        ``(delta, note)`` — ``delta`` is the largest δ on the grid where
        the median ratio is below ``c``, or ``None`` if the curve never
        drops below ``c`` on the grid. ``note`` is the empty string in
        the well-behaved case, otherwise documents the corner case.
    """
    median = jnp.median(ratios, axis=1)
    below = median < c
    if not bool(jnp.any(below)):
        return None, f"curve never below c={c}; smallest δ tested = {float(deltas[0]):.3e}"
    if bool(jnp.all(below)):
        return float(deltas[-1]), (
            f"curve stays below c={c} for every δ; reporting largest tested = "
            f"{float(deltas[-1]):.3e} as a lower bound on δ_leaf(c={c})"
        )
    idx = int(jnp.where(below, jnp.arange(deltas.shape[0]), -1).max())
    return float(deltas[idx]), ""
