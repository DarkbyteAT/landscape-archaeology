"""Behaviour tests for ``robustness``.

Each test pins one structural property of the noise-probe primitives.
Following the house style: numbers reported; structural invariants
asserted; tolerances derived from ``jnp.finfo(dtype).eps`` where the
underlying computation is float-precision, not statistics.
"""

import jax
import jax.numpy as jnp
import pytest

import landscape_archaeology as la


# --- retraining_noise_floor -------------------------------------------------
@pytest.mark.unit
def test_retraining_noise_floor_matches_sample_std_on_known_scalars():
    # Given: a train_fn that ignores the seed and returns a fixed sequence
    # via a closure over a counter (this is the cleanest way to pin σ_run
    # to a value that's hand-computable without depending on RNG).
    scalars = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    counter = {"i": 0}

    def train_fn(seed: int) -> jax.Array:
        del seed
        out = scalars[counter["i"]]
        counter["i"] += 1
        return out

    # When: we compute the noise floor across the full sequence.
    sigma = la.retraining_noise_floor(train_fn, K=5)

    # Then: it equals jnp.std(scalars, ddof=1) to float precision.
    expected = jnp.std(scalars, ddof=1)
    eps = jnp.finfo(scalars.dtype).eps
    assert jnp.allclose(sigma, expected, atol=10 * eps)


@pytest.mark.unit
def test_retraining_noise_floor_uses_independent_seeds():
    # Given: a train_fn that returns its seed as a float (so a sequence
    # of seeds produces a known sequence of scalars).
    def train_fn(seed: int) -> jax.Array:
        return jnp.asarray(float(seed))

    # When: we compute σ_run with a non-zero seed_base.
    sigma_base0 = la.retraining_noise_floor(train_fn, K=5, seed_base=0)
    sigma_base100 = la.retraining_noise_floor(train_fn, K=5, seed_base=100)

    # Then: the std of {0,1,2,3,4} equals the std of {100,...,104} since
    # std is translation-invariant; the implementation correctly passes
    # seed_base + k rather than always seed_base.
    assert jnp.allclose(sigma_base0, sigma_base100)
    # And the value matches std of an arithmetic sequence of length 5.
    expected = jnp.std(jnp.array([0.0, 1.0, 2.0, 3.0, 4.0]), ddof=1)
    assert jnp.allclose(sigma_base0, expected)


@pytest.mark.unit
def test_retraining_noise_floor_rejects_K_below_2():
    # Given: an arbitrary train_fn.
    def train_fn(seed: int) -> jax.Array:
        return jnp.asarray(0.0)

    # When/Then: K=1 raises (sample std is undefined).
    with pytest.raises(ValueError, match="K must be >= 2"):
        la.retraining_noise_floor(train_fn, K=1)


# --- perturbation_curve -----------------------------------------------------
@pytest.mark.unit
def test_perturbation_curve_returns_zero_at_delta_zero():
    # Given: a parameter pytree and an eval_fn that returns a value
    # depending on the params (any reasonable function will do).
    params = {"w": jnp.array([[1.0, 2.0], [3.0, 4.0]]), "b": jnp.array([0.5, -0.5])}

    def eval_fn(p: dict) -> jax.Array:
        return jnp.sum(p["w"] ** 2) + jnp.sum(p["b"] ** 2)

    target = eval_fn(params)
    deltas = jnp.array([0.0, 0.0, 0.0])
    key = jax.random.key(0)

    # When: we probe at δ=0 (no perturbation should ever modify the params).
    ratios = la.perturbation_curve(
        params,
        eval_fn,
        deltas,
        K_perturb=4,
        key=key,
    )

    # Then: every (δ, seed) cell equals the unperturbed eval, to float
    # precision (the multiplicative δ × frob × N(0,I) term is exactly 0).
    eps = jnp.finfo(jnp.float32).eps
    n_terms = sum(int(jnp.size(p)) for p in params.values())
    assert jnp.allclose(ratios, target, atol=10 * n_terms * eps)


@pytest.mark.unit
def test_perturbation_curve_response_grows_with_delta():
    # Given: a smooth eval_fn (sum of squares), so perturbation response
    # is monotonically larger in expectation as δ grows.
    params = {
        "w": jax.random.normal(jax.random.key(0), (8, 8)),
        "b": jax.random.normal(jax.random.key(1), (8,)),
    }

    def eval_fn(p: dict) -> jax.Array:
        return jnp.sum(p["w"] ** 2) + jnp.sum(p["b"] ** 2)

    target = eval_fn(params)
    deltas = jnp.array([1e-3, 1e-2, 1e-1])
    key = jax.random.key(42)

    # When: we probe across δ values with K seeds each.
    ratios = la.perturbation_curve(
        params,
        eval_fn,
        deltas,
        K_perturb=16,
        key=key,
    )
    delta_eval = ratios - target  # response above target
    median_response = jnp.median(jnp.abs(delta_eval), axis=1)

    # Then: median |Δeval| is strictly increasing along δ.
    assert bool(median_response[0] < median_response[1])
    assert bool(median_response[1] < median_response[2])


@pytest.mark.unit
def test_perturbation_curve_independent_seeds_give_distinct_perturbations():
    # Given: a small param pytree and an eval_fn that's sensitive to the
    # full pattern of perturbation (sum of params; signed).
    params = {"w": jnp.ones((4, 4)), "b": jnp.zeros((4,))}

    def eval_fn(p: dict) -> jax.Array:
        return jnp.sum(p["w"]) + jnp.sum(p["b"])

    deltas = jnp.array([0.1])
    key = jax.random.key(7)

    # When: we draw K=8 perturbations at the same δ.
    ratios = la.perturbation_curve(
        params,
        eval_fn,
        deltas,
        K_perturb=8,
        key=key,
    )

    # Then: every seed produces a distinct value (no accidental key
    # collision; fold_in(key, i*K+j) gives independent draws).
    values = jnp.asarray(ratios[0])
    n_unique = int(jnp.unique(values).shape[0])
    assert n_unique == 8


@pytest.mark.unit
def test_perturbation_curve_filter_freezes_unselected_leaves():
    # Given: params with two leaves; an eval_fn that depends on b only.
    params = {"w": jnp.ones((4, 4)), "b": jnp.array([1.0, 2.0, 3.0, 4.0])}

    def eval_fn(p: dict) -> jax.Array:
        # b-only response: w changes do not propagate.
        return jnp.sum(p["b"] ** 2)

    target = eval_fn(params)
    deltas = jnp.array([1e-1])
    key = jax.random.key(11)

    # When: we restrict the perturbation to leaf "w" only (so b stays at
    # the trained value and the b-only eval should be invariant).
    ratios = la.perturbation_curve(
        params,
        eval_fn,
        deltas,
        K_perturb=8,
        key=key,
        perturb_filter=lambda name: name == "w",
    )

    # Then: every cell equals the unperturbed eval (b is frozen).
    eps = jnp.finfo(jnp.float32).eps
    assert jnp.allclose(ratios, target, atol=100 * eps)


@pytest.mark.unit
def test_perturbation_curve_rejects_unknown_perturbation_kind():
    # Given: minimal valid inputs.
    params = {"w": jnp.ones((2, 2))}

    def eval_fn(p: dict) -> jax.Array:
        return jnp.sum(p["w"])

    # When/Then: an unsupported kind raises NotImplementedError.
    with pytest.raises(NotImplementedError, match="not implemented"):
        la.perturbation_curve(
            params,
            eval_fn,
            jnp.array([0.1]),
            K_perturb=1,
            key=jax.random.key(0),
            perturbation_kind="uniform",  # type: ignore[arg-type]
        )


# --- delta_from_curve -------------------------------------------------------
@pytest.mark.unit
def test_delta_from_curve_finds_crossing_at_expected_index():
    # Given: a constructed ratios matrix whose median rises through c=1.
    deltas = jnp.array([1e-3, 1e-2, 1e-1, 1e0])
    # Median across axis=1 will be [0.1, 0.5, 1.5, 5.0]; crosses c=1
    # between index 1 and 2, so δ_leaf(c=1) is deltas[1] = 1e-2.
    ratios = jnp.array(
        [
            [0.1, 0.1],
            [0.5, 0.5],
            [1.5, 1.5],
            [5.0, 5.0],
        ]
    )

    # When: we derive δ_leaf at c=1.
    delta, note = la.delta_from_curve(deltas, ratios, c=1.0)

    # Then: it equals the largest δ below the crossing.
    assert delta == pytest.approx(1e-2)
    assert note == ""


@pytest.mark.unit
def test_delta_from_curve_reports_never_crosses_corner():
    # Given: a curve that is above c=1 everywhere on the grid.
    deltas = jnp.array([1e-3, 1e-2, 1e-1])
    ratios = jnp.array([[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])

    # When: we derive δ_leaf at c=1.
    delta, note = la.delta_from_curve(deltas, ratios, c=1.0)

    # Then: delta is None and the note documents the corner.
    assert delta is None
    assert "never below" in note


@pytest.mark.unit
def test_delta_from_curve_reports_always_below_corner():
    # Given: a curve that stays below c=1 across the whole grid.
    deltas = jnp.array([1e-3, 1e-2, 1e-1])
    ratios = jnp.array([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]])

    # When: we derive δ_leaf at c=1.
    delta, note = la.delta_from_curve(deltas, ratios, c=1.0)

    # Then: delta equals the largest δ tested and the note documents the
    # corner (sweep range is too tight).
    assert delta == pytest.approx(1e-1)
    assert "stays below" in note
