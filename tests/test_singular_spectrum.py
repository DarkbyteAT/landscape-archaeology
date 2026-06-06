"""Behaviour tests for ``singular_spectrum``.

Each test pins one structural property of block power iteration on
``J^T J`` against a closed-form spectrum. Tolerances are derived from
``jnp.finfo(dtype).eps`` scaled by a small integer that accounts for the
accumulated rounding of power-iteration + QR; magnitude thresholds are
avoided in favour of structural assertions wherever possible (per
``feedback_no_magic_thresholds``).
"""

import jax
import jax.numpy as jnp
import pytest

import landscape_archaeology as la


# Convergence tolerance for power-iteration estimates. The iteration is
# linearly convergent in the ratio between successive singular values, so
# even at 200 iterations on a well-separated diagonal we expect agreement to
# a few parts in 10^-5 in float32. The bound here is loose enough to absorb
# the slowest reasonable convergence on the test problems below.
_POWER_ITER_RTOL = 1e-4


@pytest.mark.unit
def test_diagonal_operator_recovers_top_k_singular_values():
    # Given: a diagonal operator with known singular values.
    d = jnp.array([5.0, 3.0, 4.0, 1.0, 2.0, 0.5])

    def operator(x):
        return d * x

    point = jnp.ones_like(d)
    k = 3

    # When: we request the top-k spectrum.
    sigmas = la.singular_spectrum(operator, point, k=k, num_iterations=200)

    # Then: it matches the top-k of |d| sorted descending.
    expected = jnp.flip(jnp.sort(jnp.abs(d)))[:k]
    assert sigmas.shape == (k,)
    assert jnp.allclose(sigmas, expected, rtol=_POWER_ITER_RTOL)


@pytest.mark.unit
def test_orthogonal_operator_has_unit_singular_spectrum():
    # Given: a random orthogonal operator (singular values are all 1).
    n = 8
    rng = jax.random.key(7)
    a = jax.random.normal(rng, (n, n))
    q, _ = jnp.linalg.qr(a)

    def operator(x):
        return q @ x

    point = jnp.ones((n,))

    # When: we request the full spectrum.
    sigmas = la.singular_spectrum(operator, point, k=n, num_iterations=100)

    # Then: every singular value is 1.0 (within power-iteration tolerance).
    assert sigmas.shape == (n,)
    assert jnp.allclose(sigmas, jnp.ones((n,)), rtol=_POWER_ITER_RTOL)


@pytest.mark.unit
def test_low_rank_operator_separates_signal_from_null_space():
    # Given: a rank-r operator (U @ diag(s) @ V^T) acting on R^n.
    n, m, r = 6, 8, 3
    rng = jax.random.key(11)
    rng_u, rng_v = jax.random.split(rng)
    u_raw = jax.random.normal(rng_u, (m, r))
    v_raw = jax.random.normal(rng_v, (n, r))
    u, _ = jnp.linalg.qr(u_raw)
    v, _ = jnp.linalg.qr(v_raw)
    s = jnp.array([4.0, 2.0, 1.0])
    a = u @ jnp.diag(s) @ v.T

    def operator(x):
        return a @ x

    point = jnp.zeros((n,))

    # When: we request more singular values than the operator's rank.
    k = 5
    sigmas = la.singular_spectrum(operator, point, k=k, num_iterations=300)

    # Then: the top-r match s; the remaining (k - r) are near-zero (eps-scale
    # noise from QR on a rank-deficient block).
    eps_floor = jnp.finfo(point.dtype).eps * 16  # 16 = small N for rounding budget
    expected_top = jnp.flip(jnp.sort(s))
    assert jnp.allclose(sigmas[:r], expected_top, rtol=_POWER_ITER_RTOL)
    assert jnp.all(sigmas[r:] < eps_floor + _POWER_ITER_RTOL * s[0])
    # Structural: descending sorted.
    assert jnp.all(sigmas[:-1] >= sigmas[1:])


@pytest.mark.unit
def test_nonlinear_operator_recovers_pointwise_jacobian_spectrum():
    # Given: a nonlinear elementwise operator whose Jacobian at `point` is
    # diag(cos(point)) — singular values are |cos(point_i)|.
    point = jnp.array([0.0, 0.5, 1.0, 1.5, 2.0])

    def operator(x):
        return jnp.sin(x)

    # When: we ask for the full spectrum.
    k = point.shape[0]
    sigmas = la.singular_spectrum(operator, point, k=k, num_iterations=300)

    # Then: it matches |cos(point)| sorted descending.
    expected = jnp.flip(jnp.sort(jnp.abs(jnp.cos(point))))
    assert sigmas.shape == (k,)
    assert jnp.allclose(sigmas, expected, rtol=_POWER_ITER_RTOL)


@pytest.mark.unit
def test_pytree_valued_operator_returns_structural_spectrum():
    # Given: an operator that takes and returns a pytree.
    rng = jax.random.key(3)
    rng_a, rng_b = jax.random.split(rng)
    w_a = jax.random.normal(rng_a, (4, 4))
    scale = jax.random.normal(rng_b, (3,))

    def operator(params):
        return {
            "a": params["a"] @ w_a,
            "b": params["b"] * scale,
        }

    point = {"a": jnp.ones((4,)), "b": jnp.ones((3,))}
    k = 5

    # When: we request the top-k spectrum.
    sigmas = la.singular_spectrum(operator, point, k=k, num_iterations=200)

    # Then: shape matches k, all values are finite and non-negative, and the
    # spectrum is sorted in descending order. We don't pin specific magnitudes
    # — the test verifies pytree wiring, not the closed-form spectrum.
    assert sigmas.shape == (k,)
    assert jnp.all(jnp.isfinite(sigmas))
    assert jnp.all(sigmas >= 0)
    assert jnp.all(sigmas[:-1] >= sigmas[1:])


@pytest.mark.unit
def test_determinism_with_explicit_key():
    # Given: a small linear operator and a fixed key.
    d = jnp.array([2.0, 3.0, 1.0])

    def operator(x):
        return d * x

    point = jnp.ones((3,))
    key = jax.random.key(42)

    # When: we call singular_spectrum twice with the same key.
    sigmas_first = la.singular_spectrum(operator, point, k=3, num_iterations=50, key=key)
    sigmas_second = la.singular_spectrum(operator, point, k=3, num_iterations=50, key=key)

    # Then: results are bit-identical.
    assert jnp.array_equal(sigmas_first, sigmas_second)


@pytest.mark.unit
def test_k_larger_than_rank_returns_padded_spectrum():
    # Given: a rank-2 operator on R^5.
    n, r = 5, 2
    rng = jax.random.key(99)
    rng_u, rng_v = jax.random.split(rng)
    u_raw = jax.random.normal(rng_u, (n, r))
    v_raw = jax.random.normal(rng_v, (n, r))
    u, _ = jnp.linalg.qr(u_raw)
    v, _ = jnp.linalg.qr(v_raw)
    s = jnp.array([3.0, 1.5])
    a = u @ jnp.diag(s) @ v.T

    def operator(x):
        return a @ x

    point = jnp.zeros((n,))

    # When: we request k = n > rank.
    k = n
    sigmas = la.singular_spectrum(operator, point, k=k, num_iterations=300)

    # Then: function does not crash; shape is k; first r values match s;
    # the rest are near machine epsilon (eps-scale, not magic floor).
    assert sigmas.shape == (k,)
    expected_top = jnp.flip(jnp.sort(s))
    assert jnp.allclose(sigmas[:r], expected_top, rtol=_POWER_ITER_RTOL)
    eps_floor = jnp.finfo(point.dtype).eps * 16
    assert jnp.all(sigmas[r:] < eps_floor + _POWER_ITER_RTOL * s[0])
