from collections import Counter
import numpy as np
from scripts.common import load_config
from scripts.data_loading import load_observation, settled_mask_from_switch_times

FILE_PATH = "/raid1/rhino/obs_data/rhino-data/data/2026-07-11_20-22-16_obs.hdf5"
config = load_config("/raid1/rhino/rhino_vis/config.yaml")
data = load_observation(FILE_PATH, config)
settled = settled_mask_from_switch_times(data.timestamps, data.switch_times, config["analysis"]["settling_seconds"])

print("Waterfall shape :", data.waterfall.shape)
print("Frequency shape :", data.frequencies_mhz.shape)
print("Timestamp shape :", data.timestamps.shape)
print("State shape     :", data.states.shape)
print("Switch records  :", data.switch_times.shape)
print("Frequency range :", np.nanmin(data.frequencies_mhz), "to", np.nanmax(data.frequencies_mhz), "MHz")
print("\nState counts:")
for state, count in sorted(Counter(data.states).items()):
    print(f"{state:20s} {count:6d}")
print("\nSettled state counts:")
for state in sorted(set(data.states)):
    print(f"{state:20s} {np.sum((data.states == state) & settled):6d}")
print("\nSettled fraction:", np.mean(settled))
