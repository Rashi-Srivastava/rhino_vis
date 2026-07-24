from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import h5py
import numpy as np

@dataclass
class ObservationData:
    file_path: Path
    waterfall: np.ndarray
    frequencies_mhz: np.ndarray
    timestamps: np.ndarray
    states: np.ndarray
    raw_switch_states: np.ndarray
    switch_times: np.ndarray
    temperatures: np.ndarray | None = None
    temperature_timestamps: np.ndarray | None = None
    adc_i_max: np.ndarray | None = None
    adc_q_max: np.ndarray | None = None

def _read(handle, path, required=False):
    if not path:
        if required:
            raise KeyError("Required dataset path is missing.")
        return None
    key = str(path).lstrip("/")
    if key not in handle:
        if required:
            raise KeyError(f"Dataset not found: /{key}")
        return None
    return handle[key][()]

def _decode(values):
    arr = np.asarray(values)
    out = [v.decode("utf-8", errors="replace") if isinstance(v, (bytes, np.bytes_)) else str(v)
           for v in arr.reshape(-1)]
    return np.asarray(out, dtype=str).reshape(arr.shape)

def _freq_to_mhz(values):
    freq = np.asarray(values, dtype=float).squeeze()
    med = float(np.nanmedian(freq))
    if med > 1e6:
        return freq / 1e6
    if med > 1e3:
        return freq / 1e3
    return freq

def map_switch_states_to_spectra(sdr_times, switch_times, switch_states):
    sdr_times = np.asarray(sdr_times, dtype=float).reshape(-1)
    switch_times = np.asarray(switch_times, dtype=float).reshape(-1)
    switch_states = _decode(switch_states).reshape(-1)
    if switch_times.size != switch_states.size:
        raise ValueError("switch_times and switch_states lengths differ.")
    order = np.argsort(switch_times)
    switch_times = switch_times[order]
    switch_states = switch_states[order]
    idx = np.searchsorted(switch_times, sdr_times, side="right") - 1
    before_first = idx < 0
    idx = np.clip(idx, 0, switch_states.size - 1)
    states = switch_states[idx].astype(object)
    states[before_first] = "unknown"
    return states.astype(str)

def load_observation(file_path: str | Path, config: dict[str, Any]) -> ObservationData:
    file_path = Path(file_path).expanduser().resolve()
    ds = config["datasets"]
    with h5py.File(file_path, "r") as f:
        waterfall = np.asarray(_read(f, ds["waterfall"], True), dtype=float)
        freqs = _freq_to_mhz(_read(f, ds["frequency"], True))
        times = np.asarray(_read(f, ds["timestamps"], True), dtype=float).reshape(-1)
        raw_states = _decode(_read(f, ds["states"], True)).reshape(-1)
        switch_times = np.asarray(_read(f, ds["switch_times"], True), dtype=float).reshape(-1)
        temperatures = _read(f, ds.get("temperatures"))
        temperature_times = _read(f, ds.get("temperature_timestamps"))
        adc_i = _read(f, ds.get("adc_i_max"))
        adc_q = _read(f, ds.get("adc_q_max"))
    if waterfall.ndim != 2:
        raise ValueError(f"Waterfall must be 2-D, got {waterfall.shape}")
    if waterfall.shape[1] != freqs.size and waterfall.shape[0] == freqs.size:
        waterfall = waterfall.T
    if waterfall.shape != (times.size, freqs.size):
        raise ValueError(f"Shape mismatch: waterfall={waterfall.shape}, times={times.size}, freqs={freqs.size}")
    states = map_switch_states_to_spectra(times, switch_times, raw_states)
    return ObservationData(
        file_path=file_path,
        waterfall=waterfall,
        frequencies_mhz=freqs,
        timestamps=times,
        states=states,
        raw_switch_states=raw_states,
        switch_times=switch_times,
        temperatures=None if temperatures is None else np.asarray(temperatures),
        temperature_timestamps=None if temperature_times is None else np.asarray(temperature_times),
        adc_i_max=None if adc_i is None else np.asarray(adc_i),
        adc_q_max=None if adc_q is None else np.asarray(adc_q),
    )

def settled_mask_from_switch_times(sdr_times, switch_times, settling_seconds):
    sdr_times = np.asarray(sdr_times, dtype=float).reshape(-1)
    switch_times = np.asarray(switch_times, dtype=float).reshape(-1)
    idx = np.searchsorted(switch_times, sdr_times, side="right") - 1
    valid = idx >= 0
    elapsed = np.full(sdr_times.size, np.nan)
    elapsed[valid] = sdr_times[valid] - switch_times[idx[valid]]
    return valid & (elapsed >= float(settling_seconds))
