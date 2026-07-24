#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path

import numpy as np

from scripts.common import (
    configure_logging,
    ensure_observation_dirs,
    load_config,
    parse_observation_filename,
    write_json,
)

from scripts.data_loading import (
    load_observation,
    settled_mask_from_switch_times,
)

from scripts.rfi_cleaning import (
    clean_states_with_momentrfi,
)

from scripts.plotting import (
    plot_adc,
    plot_adc_by_state,
    plot_antenna_cleaning_effect,
    plot_antenna_median,
    plot_calibration_state_medians,
    plot_noise_diode,
    plot_normalised_state_medians,
    plot_raw_minus_cleaned,
    plot_state_counts,
    plot_state_waterfalls,
    plot_waterfall,
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "file",
        type=Path,
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "/raid1/rhino/rhino_vis/config.yaml"
        ),
    )

    args = parser.parse_args()

    config = load_config(
        args.config,
    )

    file_path = args.file.resolve()

    obs = parse_observation_filename(
        file_path,
        config["files"]["filename_regex"],
    )

    dirs = ensure_observation_dirs(
        config,
        obs,
    )

    configure_logging(
        Path(config["paths"]["logs_dir"])
        / f"{obs['date']}_{obs['time']}.log"
    )

    data = load_observation(
        file_path,
        config,
    )

    settled = settled_mask_from_switch_times(
        data.timestamps,
        data.switch_times,
        config["analysis"]["settling_seconds"],
    )

    prefix = obs["time"]
    title = obs["display_name"]

    dpi = int(
        config["plots"]["dpi"]
    )

    analysis_band = (
        config["analysis"]["analysis_band_mhz"]["minimum"],
        config["analysis"]["analysis_band_mhz"]["maximum"],
    )

    plot_waterfall(
        data.waterfall,
        data.frequencies_mhz,
        data.timestamps,
        f"Raw waterfall — {title}",
        dirs["raw"]
        / f"{prefix}_raw_waterfall.png",
        dpi=dpi,
        percentile_limits=(2.0, 98.0),
    )

    plot_state_waterfalls(
        data.waterfall,
        data.frequencies_mhz,
        data.timestamps,
        data.states,
        settled,
        dirs["raw"],
        prefix,
        f"Raw state waterfall — {title}",
        dpi=dpi,
    )

    plot_antenna_median(
        data.waterfall,
        data.frequencies_mhz,
        data.states,
        settled,
        f"Raw antenna median — {title}",
        dirs["raw"]
        / f"{prefix}_raw_antenna_median.png",
        dpi=dpi,
    )

    plot_calibration_state_medians(
        data.waterfall,
        data.frequencies_mhz,
        data.states,
        settled,
        f"Raw calibration-state medians — {title}",
        dirs["raw"]
        / f"{prefix}_raw_calibration_medians.png",
        dpi=dpi,
    )

    plot_normalised_state_medians(
        data.waterfall,
        data.frequencies_mhz,
        data.states,
        settled,
        f"Raw normalised state medians — {title}",
        dirs["raw"]
        / f"{prefix}_raw_normalised_state_medians.png",
        analysis_band,
        dpi=dpi,
    )

    plot_state_counts(
        data.states,
        settled,
        f"Settled spectra by state — {title}",
        dirs["states"]
        / f"{prefix}_state_counts.png",
        dpi=dpi,
    )

    plot_adc(
        data.timestamps,
        data.adc_i_max,
        data.adc_q_max,
        f"ADC maxima — {title}",
        dirs["diagnostics"]
        / f"{prefix}_adc_levels.png",
        dpi=dpi,
    )

    plot_adc_by_state(
        data.timestamps,
        data.adc_i_max,
        data.adc_q_max,
        data.states,
        f"ADC maxima by state — {title}",
        dirs["diagnostics"]
        / f"{prefix}_adc_by_state.png",
        dpi=dpi,
    )

    rfi = clean_states_with_momentrfi(
        data.waterfall,
        data.states,
        settled,
        data.frequencies_mhz,
        config,
    )

    cleaned = rfi["cleaned_waterfall"]
    mask = rfi["combined_mask"]

    if config["momentrfi"]["enabled"]:
        cleaned_title = (
            f"Cleaned waterfall — {title}"
        )
    else:
        cleaned_title = (
            f"Uncleaned waterfall "
            f"(MomentRFI disabled) — {title}"
        )

    plot_waterfall(
        cleaned,
        data.frequencies_mhz,
        data.timestamps,
        cleaned_title,
        dirs["cleaned"]
        / f"{prefix}_cleaned_waterfall.png",
        dpi=dpi,
        mask=mask if np.any(mask) else None,
        percentile_limits=(2.0, 98.0),
    )

    plot_antenna_median(
        cleaned,
        data.frequencies_mhz,
        data.states,
        settled,
        f"Cleaned antenna median — {title}",
        dirs["cleaned"]
        / f"{prefix}_cleaned_antenna_median.png",
        dpi=dpi,
    )

    plot_calibration_state_medians(
        cleaned,
        data.frequencies_mhz,
        data.states,
        settled,
        f"Cleaned calibration-state medians — {title}",
        dirs["cleaned"]
        / f"{prefix}_cleaned_calibration_medians.png",
        dpi=dpi,
    )

    plot_normalised_state_medians(
        cleaned,
        data.frequencies_mhz,
        data.states,
        settled,
        f"Cleaned normalised state medians — {title}",
        dirs["cleaned"]
        / f"{prefix}_cleaned_normalised_state_medians.png",
        analysis_band,
        dpi=dpi,
    )

    plot_raw_minus_cleaned(
        data.waterfall,
        cleaned,
        data.frequencies_mhz,
        data.timestamps,
        f"Raw minus cleaned — {title}",
        dirs["cleaned"]
        / f"{prefix}_raw_minus_cleaned.png",
        dpi=dpi,
    )

    plot_antenna_cleaning_effect(
        data.waterfall,
        cleaned,
        data.frequencies_mhz,
        data.states,
        settled,
        f"MomentRFI effect on antenna median — {title}",
        dirs["cleaned"]
        / f"{prefix}_antenna_cleaning_effect.png",
        dpi=dpi,
    )

    plot_noise_diode(
        cleaned,
        data.frequencies_mhz,
        data.states,
        settled,
        f"Noise-diode diagnostics — {title}",
        dirs["noise"]
        / f"{prefix}_noise_diode_diagnostics.png",
        dpi=dpi,
    )

    adc_i_peak = None
    adc_q_peak = None
    adc_i_over_095 = None
    adc_q_over_095 = None

    if data.adc_i_max is not None:
        adc_i_peak = float(
            np.nanmax(data.adc_i_max)
        )

        adc_i_over_095 = float(
            np.mean(
                np.asarray(data.adc_i_max)
                > 0.95
            )
        )

    if data.adc_q_max is not None:
        adc_q_peak = float(
            np.nanmax(data.adc_q_max)
        )

        adc_q_over_095 = float(
            np.mean(
                np.asarray(data.adc_q_max)
                > 0.95
            )
        )

    summary = {
        "file": str(file_path),
        "observation": title,
        "n_spectra": int(
            data.waterfall.shape[0]
        ),
        "n_channels": int(
            data.waterfall.shape[1]
        ),
        "frequency_min_mhz": float(
            np.nanmin(
                data.frequencies_mhz
            )
        ),
        "frequency_max_mhz": float(
            np.nanmax(
                data.frequencies_mhz
            )
        ),
        "n_switch_transitions": int(
            data.switch_times.size
        ),
        "settled_fraction": float(
            np.mean(settled)
        ),
        "state_counts": {
            state: int(
                np.sum(
                    (data.states == state)
                    & settled
                )
            )
            for state in sorted(
                set(data.states.tolist())
            )
        },
        "rfi_masked_fraction": float(
            np.mean(mask)
        ),
        "momentrfi_enabled": bool(
            config["momentrfi"]["enabled"]
        ),
        "adc_i_peak": adc_i_peak,
        "adc_q_peak": adc_q_peak,
        "adc_i_fraction_over_0.95": adc_i_over_095,
        "adc_q_fraction_over_0.95": adc_q_over_095,
    }

    write_json(
        dirs["outputs_day"]
        / f"{prefix}_summary.json",
        summary,
    )

    logging.info(
        "Finished successfully."
    )


if __name__ == "__main__":
    main()