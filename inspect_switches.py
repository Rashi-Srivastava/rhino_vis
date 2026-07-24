import h5py

file_path = (
    "/raid1/rhino/obs_data/rhino-data/data/"
    "2026-07-11_20-22-16_obs.hdf5"
)

with h5py.File(file_path, "r") as f:
    print("Datasets inside /switches:\n")

    for name, obj in f["switches"].items():
        print(
            f"{name:30s} "
            f"shape={obj.shape} "
            f"dtype={obj.dtype}"
        )

        values = obj[()]
        flat = values.reshape(-1)

        print("  first values:", flat[:10])
        print()