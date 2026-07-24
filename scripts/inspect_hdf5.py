#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def summarise_dataset(name: str, obj: h5py.Dataset) -> None:
    dtype = obj.dtype
    shape = obj.shape
    print(f"{name}")
    print(f"  shape : {shape}")
    print(f"  dtype : {dtype}")

    if obj.size == 0:
        print("  sample: <empty>")
        return

    try:
        sample = obj[()]
        arr = np.asarray(sample)

        if arr.ndim == 0:
            preview = repr(arr.item())
        else:
            flat = arr.reshape(-1)
            preview = repr(flat[: min(6, flat.size)].tolist())

        print(f"  sample: {preview}")

        if np.issubdtype(arr.dtype, np.number):
            finite = np.isfinite(arr)
            if finite.any():
                print(f"  finite min/max: {np.nanmin(arr):.6g} / {np.nanmax(arr):.6g}")
    except Exception as exc:
        print(f"  sample: <could not read: {exc}>")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the layout of a RHINO HDF5 observation.")
    parser.add_argument("file", type=Path)
    args = parser.parse_args()

    file_path = args.file.expanduser().resolve()
    if not file_path.is_file():
        raise FileNotFoundError(file_path)

    print(f"File: {file_path}")
    print(f"Size: {file_path.stat().st_size / 1024**2:.2f} MiB")
    print("=" * 80)

    with h5py.File(file_path, "r") as handle:
        def visitor(name, obj):
            if isinstance(obj, h5py.Group):
                print(f"[GROUP]   /{name}")
            elif isinstance(obj, h5py.Dataset):
                print(f"[DATASET] /{name}")
                summarise_dataset(f"/{name}", obj)
                print("-" * 80)

        handle.visititems(visitor)


if __name__ == "__main__":
    main()
