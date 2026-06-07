"""Behaviour tests for the closed-form spectral diagnostics.

``ht_sr_alpha`` and ``radial_fft_alpha`` both fit a log-log line to a
spectrum and return ``(alpha, r2)``. Tests pin one structural property
per scenario: known-power-law inputs recover the expected α, isotropic
spectra hit α ≈ 0, rank-deficient or degenerate inputs return NaN.
"""

import jax.numpy as jnp
import numpy as np
import pytest

import landscape_archaeology as la


# --- ht_sr_alpha ------------------------------------------------------------
@pytest.mark.unit
def test_ht_sr_alpha_recovers_synthetic_power_law():
    # Given: a matrix whose Gram eigenvalue spectrum follows a known power
    # law eigs[rank] = rank^{-target_alpha}.
    target_alpha = 1.5
    n = 64
    ranks = jnp.arange(1, n + 1).astype(jnp.float32)
    desired_eigs = ranks ** (-target_alpha)
    # Build W as a diagonal matrix whose singular-squared spectrum is desired_eigs.
    W = jnp.diag(jnp.sqrt(desired_eigs))

    # When: we fit α to its ESD.
    alpha, r2 = la.ht_sr_alpha(W)

    # Then: α recovers the input slope within a few percent and r2 ≈ 1.
    assert alpha == pytest.approx(target_alpha, rel=0.05)
    assert r2 > 0.99


@pytest.mark.unit
def test_ht_sr_alpha_flat_spectrum_returns_alpha_near_zero():
    # Given: an orthogonal-like matrix (all singular values equal).
    n = 32
    W = jnp.eye(n)

    # When: we fit α to its ESD.
    alpha, r2 = la.ht_sr_alpha(W)

    # Then: α is near zero — log-log fit of a flat spectrum has zero slope.
    assert abs(alpha) < 1e-3
    # r2 is degenerate for a flat spectrum (ss_tot ≈ 0); accept any value.
    del r2


@pytest.mark.unit
def test_ht_sr_alpha_returns_nan_on_rank_deficient_matrix():
    # Given: a zero matrix (rank 0, no surviving eigenvalues above eps).
    W = jnp.zeros((10, 10))

    # When: we fit α.
    alpha, r2 = la.ht_sr_alpha(W)

    # Then: both outputs are NaN.
    assert jnp.isnan(alpha) and jnp.isnan(r2)


@pytest.mark.unit
def test_ht_sr_alpha_handles_tall_and_wide_matrices_consistently():
    # Given: a matrix W and its transpose W.T. The ESD of W^T W has the same
    # non-zero eigenvalues as W W^T, so α and r2 should be identical (up to
    # numerical noise).
    rng = np.random.default_rng(0)
    W = jnp.asarray(rng.normal(size=(8, 32)).astype(np.float32))

    # When: we fit α on both.
    alpha_wide, r2_wide = la.ht_sr_alpha(W)
    alpha_tall, r2_tall = la.ht_sr_alpha(W.T)

    # Then: α and r2 agree to a few decimal places.
    assert alpha_wide == pytest.approx(alpha_tall, rel=1e-3)
    assert r2_wide == pytest.approx(r2_tall, rel=1e-3)


# --- radial_fft_alpha -------------------------------------------------------
@pytest.mark.unit
def test_radial_fft_alpha_returns_pair_of_floats_on_realistic_conv_kernel():
    # Given: a random conv kernel of shape (8, 3, 5, 5) — typical phase 8 shape.
    rng = np.random.default_rng(0)
    conv_w = jnp.asarray(rng.normal(size=(8, 3, 5, 5)).astype(np.float32))

    # When: we fit α to its radial-FFT spectrum.
    alpha, r2 = la.radial_fft_alpha(conv_w)

    # Then: both outputs are finite floats; r2 is in [0, 1].
    assert isinstance(alpha, float) and isinstance(r2, float)
    assert jnp.isfinite(alpha) and jnp.isfinite(r2)
    assert 0.0 <= r2 <= 1.0


@pytest.mark.unit
def test_radial_fft_alpha_returns_nan_on_zero_kernel():
    # Given: an all-zero conv kernel — no power at any frequency.
    conv_w = jnp.zeros((8, 3, 5, 5))

    # When: we fit α.
    alpha, r2 = la.radial_fft_alpha(conv_w)

    # Then: both outputs are NaN (no positive power bins to fit).
    assert jnp.isnan(alpha) and jnp.isnan(r2)


@pytest.mark.unit
def test_radial_fft_alpha_smooth_filter_has_positive_alpha():
    # Given: a smooth filter (Gaussian-shaped), pooled across channels.
    # Smooth filters concentrate energy at low frequencies, giving a
    # positive power-law exponent (α = -slope > 0).
    coords = np.linspace(-1.0, 1.0, 7)
    yy, xx = np.meshgrid(coords, coords, indexing="ij")
    gaussian = np.exp(-(yy ** 2 + xx ** 2) / 0.1).astype(np.float32)
    conv_w = jnp.asarray(np.broadcast_to(gaussian, (8, 3, 7, 7)).copy())

    # When: we fit α.
    alpha, r2 = la.radial_fft_alpha(conv_w)

    # Then: α > 0 (power decays with frequency).
    assert alpha > 0.0
    del r2  # r2 reported alongside but not asserted (depends on smoothness)


# --- Public surface --------------------------------------------------------
@pytest.mark.unit
def test_public_api_exposes_spectral_diagnostics():
    # Given: the package's public surface.
    # When: we inspect the module.
    # Then: both spectral diagnostics are exported.
    assert hasattr(la, "ht_sr_alpha")
    assert hasattr(la, "radial_fft_alpha")
    assert "ht_sr_alpha" in la.__all__
    assert "radial_fft_alpha" in la.__all__
