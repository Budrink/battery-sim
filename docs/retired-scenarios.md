# Retired demo scenarios

The 15 previous files in `examples/` were removed from the active repository
when the model switched to R1/R2/R3 and dimensionless directed connectivity.
The recovery archive and the 15 GB legacy calibration tree were permanently
deleted after the user explicitly requested freeing disk space. They cannot
be restored from this workspace.

`examples/reaction-chain-demo.json` is the only current demo; both the viewer
default and `run_viewer.sh` point to it. It is synthetic, not calibrated.
No raw experimental data were removed.

Historical Python calibration scripts and their Python tests have NOT been
removed: deleting 145 source/cache files was separately rejected pending an
explicit confirmation of that source-file scope. They are not used by the new
calibration. `upv_r123_data.py`, `upv_r123_model.py` and `calibrate_upv_r123.py`
form the independent current workflow.
File-driven CTest cases using retired demos are no longer registered; numerical
CUDA fixtures remain and the conversion fixtures now use R1/R2/R3.
