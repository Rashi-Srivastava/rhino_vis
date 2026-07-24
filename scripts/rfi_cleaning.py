from __future__ import annotations

import importlib

import numpy as np


LOAD_LIKE_STATES = {
    "load",
    "heated_load",
    "noise_diode",
    "test_src",
}

REFLECTIVE_STATES = {
    "open",
    "short",
    "long_open",
    "long_short",
}


def parameter_group_for_state(state):
    if state == "antenna":
        return "antenna"

    if state in REFLECTIVE_STATES:
        return "reflective"

    return "load_like"


def clean_states_with_momentrfi(
    waterfall,
    states,
    settled_mask,
    frequencies_mhz,
    config,
):
    cfg = config["momentrfi"]

    if not cfg.get("enabled", False):
        return {
            "cleaned_waterfall": np.asarray(
                waterfall,
                dtype=float,
            ).copy(),
            "combined_mask": np.zeros_like(
                waterfall,
                dtype=bool,
            ),
            "cleaned_by_state": {},
            "mask_by_state": {},
            "info_by_state": {},
            "indices_by_state": {},
        }

    module = importlib.import_module(
        cfg["module"]
    )

    runner = getattr(
        module,
        cfg["function"],
    )

    cleaned = np.asarray(
        waterfall,
        dtype=float,
    ).copy()

    combined_mask = np.zeros_like(
        waterfall,
        dtype=bool,
    )

    cleaned_by_state = {}
    mask_by_state = {}
    info_by_state = {}
    indices_by_state = {}

    for state in sorted(set(states.tolist())):
        idx = np.flatnonzero(
            (states == state)
            & settled_mask
        )

        if idx.size == 0:
            continue

        chunk = waterfall[idx, :]

        group = parameter_group_for_state(
            state
        )

        params = cfg["state_parameters"][
            group
        ]

        mask, wf_clean, info = runner(
            chunk,
            state_name=state,
            freqs_MHz=frequencies_mhz,

            sigma1=params["sigma1"],
            deg1_pass1=params["deg1_pass1"],
            max_iter1=params["max_iter1"],
            one_sided1=params["one_sided1"],

            sigma2=params["sigma2"],
            deg2_pass2=params["deg2_pass2"],
            max_iter2=params["max_iter2"],
            one_sided2=params["one_sided2"],
        )

        mask = np.asarray(
            mask,
            dtype=bool,
        )

        wf_clean = np.asarray(
            wf_clean,
            dtype=float,
        )

        if mask.shape != chunk.shape:
            raise ValueError(
                f"Mask shape mismatch for {state}: "
                f"{mask.shape} versus {chunk.shape}"
            )

        if wf_clean.shape != chunk.shape:
            raise ValueError(
                f"Cleaned shape mismatch for {state}: "
                f"{wf_clean.shape} versus {chunk.shape}"
            )

        cleaned[idx, :] = wf_clean
        combined_mask[idx, :] = mask

        cleaned_by_state[state] = wf_clean
        mask_by_state[state] = mask
        info_by_state[state] = info
        indices_by_state[state] = idx

    return {
        "cleaned_waterfall": cleaned,
        "combined_mask": combined_mask,
        "cleaned_by_state": cleaned_by_state,
        "mask_by_state": mask_by_state,
        "info_by_state": info_by_state,
        "indices_by_state": indices_by_state,
    }