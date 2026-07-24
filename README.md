# RHINO clean pipeline v2

Configured for the confirmed RHINO HDF5 layout and `.hdf5` filenames.

First run:

```tcsh
cd /raid1/rhino/rhino_vis
/raid1/rhino/rhino_vis/envs/rhino-vis/bin/python test_loader.py
```

Then:

```tcsh
./run_one.csh /raid1/rhino/obs_data/rhino-data/data/2026-07-11_20-22-16_obs.hdf5
```

MomentRFI is intentionally disabled for the first raw-data validation.
