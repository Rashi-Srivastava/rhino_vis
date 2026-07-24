from __future__ import annotations

import numpy as np

from MomentRFI import IterativeSurfaceFitter


def run_momentrfi_multistage(
    waterfall,
    *,
    state_name=None,
    freqs_MHz=None,
    sigma1=4.0,
    deg1_pass1=5,
    max_iter1=15,
    one_sided1=False,
    sigma2=4.0,
    deg2_pass2=10,
    max_iter2=15,
    one_sided2=False,
):
    waterfall = np.asarray(waterfall, dtype=float)

    if waterfall.ndim != 2:
        raise ValueError(
            f"Expected a 2-D waterfall, got {waterfall.shape}"
        )

    if not np.all(np.isfinite(waterfall)):
        raise ValueError(
            "MomentRFI input contains NaN or infinite values."
        )

    if np.any(waterfall <= 0):
        raise ValueError(
            "MomentRFI requires strictly positive linear-power values."
        )

    fitter = IterativeSurfaceFitter(
        sigma_threshold=float(sigma2),
        phase1_degree=int(deg1_pass1),
        phase2_degree_freq=int(deg2_pass2),
        phase2_degree_time=int(deg1_pass1),
        max_iterations=max(
            int(max_iter1),
            int(max_iter2),
        ),
        one_sided_clipping=bool(one_sided2),
        verbose=True,
    )

    mask = np.asarray(
        fitter.fit(waterfall),
        dtype=bool,
    )

    fitted_surface = np.asarray(
        fitter.surface,
        dtype=float,
    )

    cleaned = waterfall.copy()
    cleaned[mask] = fitted_surface[mask]

    info = {
        "state_name": state_name,
        "masked_fraction": float(np.mean(mask)),
        "n_masked": int(np.sum(mask)),
        "n_pixels": int(mask.size),
        "sigma_floor": (
            None
            if fitter.sigma_floor is None
            else float(fitter.sigma_floor)
        ),
        "history": fitter.history,
        "fitted_surface": fitted_surface,
        "residuals": np.asarray(
            fitter.residuals,
            dtype=float,
        ),
    }

    return mask, cleaned, info
