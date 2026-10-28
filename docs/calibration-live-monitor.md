# Live calibration monitor

The native Dear ImGui/ImPlot monitor reads the `status.json`, `experiment.json`
and `live.json` files in one calibration run directory. The calibration runner
publishes compact atomic snapshots; the viewer never drives the simulation.
Experiment channels and model cells are separate selectable curves: TC-to-cell
mapping is not known. Temperatures are Celsius, without time alignment or shifts.

After building `battery_sim_calibration_monitor`, launch it with a run directory:

```sh
build/ui/battery_sim_calibration_monitor data/development/RUN_DIRECTORY
```

The warmup and cascade prefixes are concatenated at their real checkpoint time.
The displayed cascade speed is for the post-checkpoint segment, not the entire
run. The viewer rereads data every two seconds; new points appear when the
simulation emits its next intermediate snapshot.
Older runs do not automatically contain these live snapshots.

The current 2026-10-27 diagnostic starts from the 0.2x R2 seed, a 0.25 mm
metal contact strip, original gas flow resistance, 3x gas mixing and 1.5x R3
rate. Gas mixing exchanges species as well as heat. These are unmeasured
fixture hypotheses, not calibrated material laws or independent validation.
