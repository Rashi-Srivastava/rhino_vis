import h5py

file_path = "/raid1/rhino/obs_data/rhino-data/data/2026-07-11_20-22-16_obs.hdf5"

with h5py.File(file_path, "r") as f:
    print("Datasets inside /sdr:\n")
    for name, obj in f["sdr"].items():
        print(f"{name:30s} shape={obj.shape} dtype={obj.dtype}")