from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def save_figure(fig, path, dpi=180):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(fig)

    return path


def relative_hours(times):
    times = np.asarray(times, dtype=float)

    return (times - times[0]) / 3600.0


def robust_limits(data, lower=2.0, upper=98.0):
    data = np.asarray(data, dtype=float)

    finite = np.isfinite(data)

    if not np.any(finite):
        return None, None

    vmin = np.nanpercentile(
        data[finite],
        lower,
    )

    vmax = np.nanpercentile(
        data[finite],
        upper,
    )

    if not np.isfinite(vmin) or not np.isfinite(vmax):
        return None, None

    if vmax <= vmin:
        return None, None

    return vmin, vmax


def plot_waterfall(
    waterfall,
    freqs,
    times,
    title,
    output_path,
    dpi=180,
    mask=None,
    percentile_limits=(2.0, 98.0),
):
    data = np.asarray(
        waterfall,
        dtype=float,
    )

    if mask is not None:
        data = np.ma.array(
            data,
            mask=np.asarray(mask, dtype=bool),
        )

    hours = relative_hours(times)

    lower, upper = percentile_limits

    vmin, vmax = robust_limits(
        np.asarray(data),
        lower=lower,
        upper=upper,
    )

    fig, ax = plt.subplots(
        figsize=(12, 6),
    )

    image = ax.imshow(
        data,
        aspect="auto",
        interpolation="nearest",
        extent=[
            freqs[0],
            freqs[-1],
            hours[-1],
            hours[0],
        ],
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Time since start [h]")
    ax.set_title(title)

    fig.colorbar(
        image,
        ax=ax,
        label="Power",
    )

    return save_figure(
        fig,
        output_path,
        dpi,
    )


def plot_antenna_median(
    waterfall,
    freqs,
    states,
    settled,
    title,
    output_path,
    dpi=180,
):
    idx = (
        (states == "antenna")
        & settled
    )

    if not np.any(idx):
        return None

    median = np.nanmedian(
        waterfall[idx],
        axis=0,
    )

    fig, ax = plt.subplots(
        figsize=(12, 5),
    )

    ax.plot(
        freqs,
        median,
    )

    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Median power")
    ax.set_title(title)
    ax.grid(alpha=0.25)

    return save_figure(
        fig,
        output_path,
        dpi,
    )


def plot_calibration_state_medians(
    waterfall,
    freqs,
    states,
    settled,
    title,
    output_path,
    dpi=180,
):
    calibration_states = [
        "load",
        "heated_load",
        "noise_diode",
        "open",
        "short",
        "long_open",
        "long_short",
        "test_src",
    ]

    fig, ax = plt.subplots(
        figsize=(12, 6),
    )

    plotted = False

    for state in calibration_states:
        idx = (
            (states == state)
            & settled
        )

        if not np.any(idx):
            continue

        median = np.nanmedian(
            waterfall[idx],
            axis=0,
        )

        ax.plot(
            freqs,
            median,
            label=state,
        )

        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Median power")
    ax.set_title(title)
    ax.legend(
        ncol=2,
        fontsize=8,
    )
    ax.grid(alpha=0.25)

    return save_figure(
        fig,
        output_path,
        dpi,
    )


def plot_normalised_state_medians(
    waterfall,
    freqs,
    states,
    settled,
    title,
    output_path,
    analysis_band,
    dpi=180,
):
    band = (
        (freqs >= analysis_band[0])
        & (freqs <= analysis_band[1])
    )

    fig, ax = plt.subplots(
        figsize=(12, 6),
    )

    plotted = False

    for state in sorted(set(states.tolist())):
        idx = (
            (states == state)
            & settled
        )

        if not np.any(idx):
            continue

        median = np.nanmedian(
            waterfall[idx],
            axis=0,
        )

        reference = np.nanmedian(
            median[band],
        )

        if not np.isfinite(reference) or reference == 0:
            continue

        normalised = median / reference

        ax.plot(
            freqs,
            normalised,
            label=state,
        )

        plotted = True

    if not plotted:
        plt.close(fig)
        return None

    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Power / band median")
    ax.set_title(title)
    ax.legend(
        ncol=2,
        fontsize=8,
    )
    ax.grid(alpha=0.25)

    return save_figure(
        fig,
        output_path,
        dpi,
    )


def plot_state_waterfalls(
    waterfall,
    freqs,
    times,
    states,
    settled,
    output_dir,
    prefix,
    title_prefix,
    dpi=180,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for state in sorted(set(states.tolist())):
        idx = (
            (states == state)
            & settled
        )

        if not np.any(idx):
            continue

        state_waterfall = waterfall[idx]
        state_times = times[idx]

        if state == "antenna":
            limits = (5.0, 95.0)
        else:
            limits = (2.0, 98.0)

        plot_waterfall(
            state_waterfall,
            freqs,
            state_times,
            f"{title_prefix}: {state}",
            output_dir
            / f"{prefix}_{state}_waterfall.png",
            dpi=dpi,
            percentile_limits=limits,
        )


def plot_state_counts(
    states,
    settled,
    title,
    output_path,
    dpi=180,
):
    unique = sorted(
        set(states.tolist())
    )

    counts = [
        int(
            np.sum(
                (states == state)
                & settled
            )
        )
        for state in unique
    ]

    fig, ax = plt.subplots(
        figsize=(10, 5),
    )

    ax.bar(
        unique,
        counts,
    )

    ax.set_ylabel("Settled spectra")
    ax.set_title(title)

    ax.tick_params(
        axis="x",
        rotation=35,
    )

    return save_figure(
        fig,
        output_path,
        dpi,
    )


def plot_noise_diode(
    waterfall,
    freqs,
    states,
    settled,
    title,
    output_path,
    dpi=180,
):
    ns = (
        (states == "noise_diode")
        & settled
    )

    load = (
        (states == "load")
        & settled
    )

    if not np.any(ns) or not np.any(load):
        return None

    pns = np.nanmedian(
        waterfall[ns],
        axis=0,
    )

    pload = np.nanmedian(
        waterfall[load],
        axis=0,
    )

    difference = pns - pload

    ratio = np.divide(
        pns,
        pload,
        out=np.full_like(
            pns,
            np.nan,
        ),
        where=pload != 0,
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
    )

    axes[0].plot(
        freqs,
        difference,
    )

    axes[0].set_ylabel(
        "P_NS - P_load"
    )
    axes[0].grid(alpha=0.25)

    axes[1].plot(
        freqs,
        ratio,
    )

    axes[1].set_xlabel(
        "Frequency [MHz]"
    )
    axes[1].set_ylabel(
        "P_NS / P_load"
    )
    axes[1].grid(alpha=0.25)

    fig.suptitle(title)

    return save_figure(
        fig,
        output_path,
        dpi,
    )


def plot_adc(
    times,
    adc_i,
    adc_q,
    title,
    output_path,
    dpi=180,
):
    if adc_i is None or adc_q is None:
        return None

    hours = relative_hours(times)

    fig, ax = plt.subplots(
        figsize=(12, 4),
    )

    ax.plot(
        hours,
        np.asarray(adc_i).reshape(-1),
        label="max |I|",
    )

    ax.plot(
        hours,
        np.asarray(adc_q).reshape(-1),
        label="max |Q|",
    )

    ax.axhline(
        0.95,
        linestyle="--",
        linewidth=1,
        label="0.95 threshold",
    )

    ax.set_xlabel("Time since start [h]")
    ax.set_ylabel("ADC maximum")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.25)

    return save_figure(
        fig,
        output_path,
        dpi,
    )


def plot_adc_by_state(
    times,
    adc_i,
    adc_q,
    states,
    title,
    output_path,
    dpi=180,
):
    if adc_i is None or adc_q is None:
        return None

    hours = relative_hours(times)

    adc_i = np.asarray(
        adc_i,
    ).reshape(-1)

    adc_q = np.asarray(
        adc_q,
    ).reshape(-1)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 8),
        sharex=True,
    )

    for state in sorted(set(states.tolist())):
        idx = states == state

        if not np.any(idx):
            continue

        axes[0].scatter(
            hours[idx],
            adc_i[idx],
            s=3,
            label=state,
        )

        axes[1].scatter(
            hours[idx],
            adc_q[idx],
            s=3,
            label=state,
        )

    axes[0].axhline(
        0.95,
        linestyle="--",
        linewidth=1,
    )

    axes[1].axhline(
        0.95,
        linestyle="--",
        linewidth=1,
    )

    axes[0].set_ylabel("max |I|")
    axes[1].set_ylabel("max |Q|")
    axes[1].set_xlabel(
        "Time since start [h]"
    )

    axes[0].legend(
        ncol=2,
        fontsize=8,
    )

    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)

    fig.suptitle(title)

    return save_figure(
        fig,
        output_path,
        dpi,
    )


def plot_raw_minus_cleaned(
    raw,
    cleaned,
    freqs,
    times,
    title,
    output_path,
    dpi=180,
):
    difference = (
        np.asarray(raw, dtype=float)
        - np.asarray(cleaned, dtype=float)
    )

    return plot_waterfall(
        difference,
        freqs,
        times,
        title,
        output_path,
        dpi=dpi,
        percentile_limits=(2.0, 98.0),
    )


def plot_antenna_cleaning_effect(
    raw,
    cleaned,
    freqs,
    states,
    settled,
    title,
    output_path,
    dpi=180,
):
    idx = (
        (states == "antenna")
        & settled
    )

    if not np.any(idx):
        return None

    raw_median = np.nanmedian(
        raw[idx],
        axis=0,
    )

    clean_median = np.nanmedian(
        cleaned[idx],
        axis=0,
    )

    fractional_change = np.divide(
        clean_median - raw_median,
        raw_median,
        out=np.full_like(
            raw_median,
            np.nan,
        ),
        where=raw_median != 0,
    )

    fig, ax = plt.subplots(
        figsize=(12, 4),
    )

    ax.plot(
        freqs,
        100.0 * fractional_change,
    )

    ax.axhline(
        0,
        linewidth=1,
    )

    ax.set_xlabel("Frequency [MHz]")
    ax.set_ylabel("Median change [%]")
    ax.set_title(title)
    ax.grid(alpha=0.25)

    return save_figure(
        fig,
        output_path,
        dpi,
    )