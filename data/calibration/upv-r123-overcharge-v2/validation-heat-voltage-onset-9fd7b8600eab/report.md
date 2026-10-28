# Frozen thermal validation

Only the pre-event thermal phase was checked; explosion is not calibrated.
No model parameters were changed. The same model/binary fingerprint and profile were verified before/after replay.
The original validation generator incorrectly applied 35 A during pre-charge rest. This diagnostic scenario starts current at the midpoint of the measured voltage-step bracket, not a fitted thermal shift.
Charge onset bracket: [466.17099980358034, 466.83699963614345]; midpoint 466.504000 s.
Cooling sink = 10 W/K, unchanged engineering prior; no held-out fitting.

| Channel | Training RMSE, K (charging) | Validation RMSE, K (charging) | Validation mean bias, K (charging) | Validation final error, K |
|---|---:|---:|---:|---:|
| C1 / ISS | 2.694 | 3.947 | -2.431 | -10.076 |
| C1 / TIM | 8.061 | 5.881 | +4.692 | +10.782 |
| C3 / TIM | 2.597 | 2.615 | -1.501 | -6.967 |
| C3 / ISS | 6.700 | 4.259 | +3.504 | +6.590 |
| C2 / TIM | 2.782 | 13.260 | -9.953 | -25.634 |
| C2 / ISS | 3.693 | 11.289 | -8.384 | -14.596 |

Validation duration 1392.491000 simulated seconds, wall 409.153 seconds.
Positive bias means calculated temperature is too high; negative means too low.

The source generator scripts/upv_r123_data.py was not edited in this diagnostic review; the corrected input is preserved in validate-scenario.json.
