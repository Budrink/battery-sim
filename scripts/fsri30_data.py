"""Immutable UL FSRI v1.1 input; no solver and no fitted synchronization."""
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/experiments/ul-fsri-30'
ARCHIVE_URL = 'https://codeload.github.com/ulfsri/fsri-lib-ess-demo-2020/zip/refs/tags/v1.1'
REPORT_URL = 'https://fsri.org/sites/default/files/2021-07/UL9540AInstallationDemo_Report_Final_4-12-21.pdf'
COMMIT = '8d140bc02f2e2cd04c2ca95f1d925ec4c24365b3'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def import_archive(archive, report):
    """Preserve raw bytes; reject replacement of any already archived source."""
    DATA.mkdir(parents=True, exist_ok=True)
    targets = [(Path(archive), DATA/'source-v1.1.zip'),
               (Path(report), DATA/'installation-report.pdf')]
    for source, target in targets:
        if target.exists() and sha(target) != sha(source):
            raise ValueError('Refusing to overwrite a different raw input: '+str(target))
        if not target.exists():
            shutil.copyfile(source, target); target.chmod(0o444)
    with zipfile.ZipFile(targets[0][1]) as z:
        member = next(n for n in z.namelist() if n.endswith('/Cell_Level_Experiment.csv'))
        raw = z.read(member)
        commit_note = z.comment.decode()
        if commit_note != COMMIT:
            raise ValueError('Source tag does not match pinned commit')
    result = dict(dataset_id='ulfsri/fsri-lib-ess-demo-2020/v1.1', commit=COMMIT,
        source_url=ARCHIVE_URL, doi='https://doi.org/10.5281/zenodo.7244850',
        archive_sha256=sha(targets[0][1]), report_url=REPORT_URL,
        report_sha256=sha(targets[1][1]), csv_member=member,
        csv_sha256=hashlib.sha256(raw).hexdigest(), retrieved_at='2026-10-23',
        license='Repository MIT; PDF report retained with its original attribution/rights',
        split='development', calibration_status='UNFITTED engineering reconstruction',
        temperature_sensor_locations='Nine CSV channels; exact component correspondence unconfirmed',
        timing='Published Time (s) only; no row-index extrapolation, shifts or warping')
    save(DATA/'manifest.json', result)
    return result


def load():
    manifest = json.loads((DATA/'manifest.json').read_text())
    if sha(DATA/'source-v1.1.zip') != manifest['archive_sha256']:
        raise ValueError('FSRI archive checksum changed')
    with zipfile.ZipFile(DATA/'source-v1.1.zip') as z:
        raw = z.read(manifest['csv_member'])
    if hashlib.sha256(raw).hexdigest() != manifest['csv_sha256']:
        raise ValueError('FSRI CSV checksum changed')
    rows = list(csv.DictReader(io.StringIO(raw.decode('utf-8-sig'))))
    # The file ends with temperature-only rows without physical timestamps.
    # They are NOT silently assigned an artificial time axis.
    known = [row for row in rows if row['Time (s)'].strip()]
    result = {'time_s': np.array([float(r['Time (s)']) for r in known])}
    if np.any(np.diff(result['time_s']) <= 0):
        raise ValueError('FSRI timestamps must strictly increase')
    for n in range(1, 10):
        result[f'T{n}_k'] = np.array([float(r[f'Cell {n} Temperature (C)'])+273.15
            if r[f'Cell {n} Temperature (C)'].strip() else np.nan for r in known])
    for name, col in [('hrr_kw','Heat Release Rate (kW)'), ('hydrogen_l_min','H2 Flow (L/min)')]:
        result[name] = np.array([float(r[col]) if r[col].strip() else np.nan for r in known])
    for name, col in [('runaway','Thermal Runaway'), ('flaming','Flaming')]:
        result[name] = np.array([r[col] == 'TRUE' for r in known])
    manifest = dict(manifest, untimed_rows_excluded=len(rows)-len(known),
        valid_end_s=float(result['time_s'][-1]),
        missing_temperature_samples={f'T{n}': int(np.sum(~np.isfinite(result[f'T{n}_k']))) for n in range(1,10)})
    return result, manifest
