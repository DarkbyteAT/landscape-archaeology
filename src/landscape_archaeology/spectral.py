"""Power-law diagnostics on weight-matrix and conv-kernel spectra.

Two functions, both of the same shape: fit a log-log line to a spectrum,
return ``(alpha, r2)`` where ``alpha = -slope`` is the power-law exponent
and ``r2`` is the coefficient of determination of the fit.

- :func:`ht_sr_alpha` runs on the eigenvalue spectrum of ``W^T W``
  (heavy-tailed self-regularisation, Martin & Mahoney 2018-2021), the
  standard diagnostic for fully-connected weight matrices. Empirical α
  closer to 2 indicates a well-trained heavy-tailed distribution; α far
  from 2 (large α: random matrix-like; small α: pre-power-law) flags
  under-/over-training.

- :func:`radial_fft_alpha` runs on the radial-frequency spectrum of a
  conv kernel (channels pooled, 2-D FFT magnitude radially averaged).
  Same power-law-fit shape, applied to convolution filters whose ESD is
  not informative because spatial dimensions are tiny (kernel size 3-7).

Both are "spectrum-derived diagnostics", consistent with
landscape-archaeology owning Jacobian / Hessian spectral probes alongside
the more general :func:`singular_spectrum`.
"""

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float


def ht_sr_alpha(W: Float[Array, "out in_"]) -> tuple[float, float]:
    """Heavy-tailed self-regularisation slope of ``W``'s ESD.

    Computes eigenvalues of ``W^T W`` (or ``W W^T`` when more efficient),
    fits a log-log line to ``(rank, eigenvalue)``, and returns
    ``(alpha, r2)`` where ``alpha = -slope`` and ``r2`` is the fit's
    coefficient of determination. Returns ``(nan, nan)`` when fewer than
    two non-trivial eigenvalues remain after thresholding (rank-deficient
    matrix).

    Args:
        W: a 2-D fc-weight matrix. The function uses ``W^T W`` when
            ``out >= in_`` and ``W W^T`` otherwise so the eigendecomposition
            is always on the smaller Gram matrix.

    Returns:
        ``(alpha, r2)``. ``alpha`` is the power-law exponent
        (``α = -slope``). ``r2 ∈ [0, 1]`` is the fit's coefficient of
        determination; values near 1 indicate the spectrum is well-described
        by a power law over its support.

    Notes:
        Thresholding follows the conventional choice of dropping eigenvalues
        below ``finfo.eps · n · max(top_eig, 1)``; this is more conservative
        than "exactly zero" but absorbs the floating-point noise that
        otherwise pollutes the low end of the log-log fit.
    """
    M = W.T @ W if W.shape[0] >= W.shape[1] else W @ W.T
    eigs = jnp.sort(jnp.linalg.eigvalsh(M))[::-1]
    eps = jnp.finfo(eigs.dtype).eps * eigs.shape[0] * jnp.maximum(eigs[0], jnp.array(1.0))
    mask = eigs > eps
    eigs_pos = eigs[mask]
    n = eigs_pos.shape[0]
    if n < 2:
        return float("nan"), float("nan")
    ranks = jnp.arange(1, n + 1)
    log_r = jnp.log(ranks)
    log_p = jnp.log(eigs_pos)
    slope, intercept = jnp.polyfit(log_r, log_p, 1)
    alpha = float(-slope)
    log_p_hat = slope * log_r + intercept
    ss_res = float(jnp.sum((log_p - log_p_hat) ** 2))
    ss_tot = float(jnp.sum((log_p - jnp.mean(log_p)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
    return alpha, r2


def radial_fft_alpha(conv_w: Float[Array, "out in_ kH kW"]) -> tuple[float, float]:
    """Power-law slope of a conv kernel's radial-frequency spectrum.

    Computes the 2-D FFT of each ``(out, in_)`` slice, pools squared
    magnitude over channel dims, radially averages, and fits a log-log
    line to ``(radial_freq, power)``. Returns ``(alpha, r2)`` where
    ``alpha = -slope`` and ``r2`` is the fit's coefficient of determination.

    Args:
        conv_w: a 4-D conv-kernel tensor of shape ``(out, in_, kH, kW)``.

    Returns:
        ``(alpha, r2)``. ``alpha`` is the power-law exponent
        (``α = -slope``); large α means power concentrates at low
        frequency (smooth filter); small α means flat spectrum (more
        random-noise-like).

    Notes:
        Small kernel sizes (``kH = kW ≤ 5``) under-recover the true α
        because the radial grid has few non-zero bins. Read the value
        alongside ``r2`` rather than at face value.
    """
    W_np = np.asarray(conv_w)
    _out, _in, kH, kW = W_np.shape
    F = np.fft.fft2(W_np, axes=(-2, -1))
    power = np.abs(F) ** 2
    power_pooled = power.sum(axis=(0, 1))

    ky = np.fft.fftfreq(kH) * kH
    kx = np.fft.fftfreq(kW) * kW
    KY, KX = np.meshgrid(ky, kx, indexing="ij")
    R = np.sqrt(KY ** 2 + KX ** 2)
    R_int = np.round(R).astype(int)
    radial_bins = np.arange(0, int(R_int.max()) + 1)
    radial_mean = np.array([
        power_pooled[R_int == r].mean() if (R_int == r).any() else np.nan
        for r in radial_bins
    ])
    finite = np.isfinite(radial_mean) & (radial_mean > 0) & (radial_bins > 0)
    if finite.sum() < 2:
        return float("nan"), float("nan")
    log_k = np.log(radial_bins[finite].astype(np.float64))
    log_p = np.log(radial_mean[finite].astype(np.float64))
    slope, intercept = np.polyfit(log_k, log_p, 1)
    alpha = float(-slope)
    log_p_hat = slope * log_k + intercept
    ss_res = float(np.sum((log_p - log_p_hat) ** 2))
    ss_tot = float(np.sum((log_p - np.mean(log_p)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
    return alpha, r2


__all__ = ["ht_sr_alpha", "radial_fft_alpha"]
