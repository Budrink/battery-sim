#!/usr/bin/env python3
"""Fit/check the native CUDA model against the old synthetic 52-battery reference.

Python only chooses parameters and scores events. All candidate simulations run
in battery_sim_reference_probe, using the same CUDA solver as the viewer.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]

def metrics(run, reference):
    actual, target = run["onsets"], reference["runaway_times_s"]
    ids = sorted(set(target)-{"r2-c07"})
    errors = [actual.get(k, 1800)-target[k] for k in ids]
    row = [actual.get(f"r2-c{i:02}", math.inf) for i in range(1, 14)]
    other = [v for k, v in actual.items() if not k.startswith("r2-")]
    return dict(
        reached=len(actual), rms_onset_error_s=math.sqrt(sum(e*e for e in errors)/len(errors)),
        max_onset_error_s=max(abs(e) for e in errors),
        completion_error_s=run["complete_s"]-reference["complete_s"],
        first_row_before_other_rows=bool(other) and max(row)<min(other),
        relative_energy_residual=abs(run["residual_j"])/max(run["released_j"], 1),
        per_cell_error_s=dict(zip(ids, errors)))

def acceptable(run, score):
    return (score["reached"]==52 and score["first_row_before_other_rows"]
            and score["rms_onset_error_s"]<=10 and score["max_onset_error_s"]<=25
            and abs(score["completion_error_s"])<=20
            and abs(run["onsets"].get("r2-c07", -1)-5)<1e-8
            and score["relative_energy_residual"]<1e-10)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, default=ROOT/"build/battery_sim_reference_probe")
    parser.add_argument("--fit", action="store_true", help="Refit six contact parameters; requires scipy")
    parser.add_argument("--check", action="store_true", help="Fail if the full-grid numerical regression exceeds its tolerances")
    parser.add_argument("--max-evaluations", type=int, default=12)
    parser.add_argument("--max-dt", type=float, default=.2)
    parser.add_argument("--duration", type=float, default=1400)
    parser.add_argument("--mesh-scale", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args=parser.parse_args()
    reference=json.loads((ROOT/"data/calibration/narada-lumped-reference.json").read_text())
    profile=json.loads((ROOT/"data/calibration/narada-reference-fit.json").read_text())
    scenario=ROOT/"examples"/("narada-52-multiscale-prior.json" if args.fit else "narada-52-multiscale.json")
    worker=subprocess.Popen([str(args.probe.resolve()), str(scenario)], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, text=True, bufsize=1)
    evaluations=[]
    def run(parameters):
        worker.stdin.write(json.dumps(parameters)+"\n"); worker.stdin.flush()
        line=worker.stdout.readline()
        if not line: raise RuntimeError("Native CUDA probe exited without a report")
        result=json.loads(line)
        if "error" in result: raise RuntimeError(result["error"])
        return result
    try:
        parameters=dict(profile["parameters"])
        if args.fit:
            from scipy.optimize import least_squares
            keys=["gb", "gx", "tb", "tx", "coldb", "coldx"]
            def objective(x):
                candidate=dict(parameters, **dict(zip(keys, map(float, x))))
                result=run(dict(candidate, coarse=True))
                score=metrics(result, reference); evaluations.append((candidate, result, score))
                print(f"candidate {len(evaluations)}: RMS {score['rms_onset_error_s']:.3f} s, reached {score['reached']}", flush=True)
                return list(score["per_cell_error_s"].values())
            least_squares(objective, [parameters[k] for k in keys],
                          bounds=([1.5,2,370,370,.5,.3], [8,35,490,500,2.5,2]),
                          diff_step=.002, x_scale="jac", max_nfev=args.max_evaluations,
                          ftol=.001, xtol=.0003)
            parameters=min(evaluations, key=lambda e:e[2]["rms_onset_error_s"])[0]
        request=dict(parameters, coarse=False, mesh_scale=args.mesh_scale, dt=args.max_dt) if args.fit else dict(apply_fit=False, mesh_scale=args.mesh_scale, dt=args.max_dt)
        request["duration"]=args.duration
        result=run(request); score=metrics(result, reference)
        full_window=args.duration>=max(reference["runaway_times_s"].values())+60
        accepted=acceptable(result, score) if full_window else None
        if not full_window:
            observed=[result["onsets"][k]-t for k,t in reference["runaway_times_s"].items()
                      if k!="r2-c07" and k in result["onsets"]]
            score=dict(reached=len(result["onsets"]), compared_secondary_events=len(observed),
                       rms_observed_onset_error_s=math.sqrt(sum(e*e for e in observed)/len(observed)) if observed else None,
                       max_observed_onset_error_s=max(map(abs,observed)) if observed else None,
                       relative_energy_residual=score["relative_energy_residual"])
        report=dict(kind="numerical-reference-comparison; NOT experimental validation",
                    scope="full-cascade" if full_window else "partial-window; full cascade not tested",
                    reference=profile["reference"], parameters=parameters, metrics=score,
                    run=result, accepted=accepted, fit_candidates=len(evaluations))
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps({k:v for k,v in score.items() if k!="per_cell_error_s"}, indent=2))
        print(f"complete_s={result['complete_s']:.3f}; volumes={result['volumes']}; accepted={report['accepted']}")
        return 0 if not args.check or report["accepted"] else 1
    finally:
        worker.stdin.close()
        worker.stdout.close()
        try: worker.wait(timeout=5)
        except subprocess.TimeoutExpired: worker.terminate(); worker.wait()

if __name__=="__main__":
    raise SystemExit(main())
