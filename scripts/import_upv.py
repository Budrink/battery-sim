#!/usr/bin/env python3
"""Import immutable UPV instrument exports; no simulation or parameter fitting.

Only the Python standard library is used. XLSX time values are Excel DAYS,
despite the instrument's 'Scan Sweep Time (Sec)' label. Invalid readings are
masked, never clipped to a plausible temperature. The original ZIP is retained.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = 'https://zenodo.org/records/22273467'
ARCHIVE_MD5 = '11ef7ee807ba923e1929ba2abd8445c9'
EXPERIMENTS = {
    'single-cell': ('characterization', '02_SingleCell/SingleCell_Temperature_Voltage_20250613.xlsx'),
    'no-cooling': ('calibration', '01_MiniModule_NoCooling/MiniModule_NoCooling_Temperature_Pressure_20250701.xlsx'),
    'active-cooling': ('validation', '03_MiniModule_ActiveCooling/MiniModule_Cooling_Temperature_Pressure_Voltage_20250729.xlsx'),
}
NS = {'s': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def xlsx_rows(data):
    """Sparse XLSX cells, shared/inline strings, and numeric values, first sheet."""
    with zipfile.ZipFile(io.BytesIO(data)) as book:
        strings = []
        if 'xl/sharedStrings.xml' in book.namelist():
            strings = [''.join(x.itertext()) for x in ET.fromstring(book.read('xl/sharedStrings.xml'))]
        root = ET.fromstring(book.read('xl/worksheets/sheet1.xml'))
        for row in root.findall('s:sheetData/s:row', NS):
            result = {}
            for cell in row:
                column = re.match(r'[A-Z]+', cell.attrib['r'])[0]
                value = cell.find('s:v', NS)
                text = value.text if value is not None else ''
                if cell.get('t') == 's' and text:
                    text = strings[int(text)]
                elif cell.get('t') == 'inlineStr':
                    text = ''.join(cell.find('s:is', NS).itertext())
                result[column] = text
            yield result


def temperature_quality(celsius):
    if not math.isfinite(celsius) or abs(celsius) >= 1e30:
        return 'instrument_overrange'
    # Broad physical/instrument envelope, not a fitted outlier rejection rule.
    if celsius < -100 or celsius > 1400:
        return 'outside_type_k_analysis_envelope'
    return 'valid'


def parse_export(data):
    iterator = iter(xlsx_rows(data))
    for header in iterator:
        if header.get('A') == 'Scan Sweep Time (Sec)' and header.get('B') == 'Scan Number':
            break
    else:
        raise ValueError('Instrument data header missing')
    columns = {}
    for col, label in header.items():
        match = re.fullmatch(r'(\d+) \((°C|Vdc)\)', label)
        if match:
            columns[col] = (match[1], match[2])
    if not columns:
        raise ValueError('No supported instrument channels')
    records, first_day, last_time, last_scan = [], None, -1., 0
    counts = {channel: {} for channel, _ in columns.values()}
    for row in iterator:
        if not row.get('A'):
            continue
        day, scan = float(row['A']), int(row['B'])
        if not 40000 < day < 60000:
            raise ValueError('Expected Excel serial days, not seconds')
        if first_day is None:
            first_day = day
        time = (day-first_day)*86400
        if time <= last_time or scan != last_scan+1:
            raise ValueError('Nonmonotonic time or missing/duplicate scan; refusing implicit repair')
        record = {'time_s': time, 'scan': scan}
        for col, (channel, unit) in columns.items():
            value = float(row[col]) if row.get(col) else math.nan
            quality = temperature_quality(value) if unit == '°C' else ('valid' if math.isfinite(value) and abs(value)<1e30 else 'instrument_overrange')
            counts[channel][quality] = counts[channel].get(quality, 0)+1
            key = f'T{channel}_k' if unit == '°C' else f'U{channel}_v'
            record[key] = value+273.15 if quality == 'valid' and unit == '°C' else (value if quality == 'valid' else '')
            record[f'q{channel}'] = quality
        records.append(record)
        last_time, last_scan = time, scan
    if len(records)<2:
        raise ValueError('Export has fewer than two data rows')
    metadata = dict(samples=len(records), duration_s=last_time,
                    median_dt_s=statistics.median(b['time_s']-a['time_s'] for a,b in zip(records, records[1:])),
                    channel_quality_counts=counts, excel_start_day=first_day,
                    raw_time_unit='Excel serial days (mislabeled seconds)',
                    conversions=['elapsed_s=(excel_day-first_excel_day)*86400', 'kelvin=celsius+273.15'],
                    quality_policy='Invalid values blank with explicit quality code; no interpolation or clipping; spikes are NOT automatically removed')
    return records, metadata


def import_archive(path, output):
    raw = path.read_bytes()
    if hashlib.md5(raw).hexdigest() != ARCHIVE_MD5:
        raise ValueError('ZIP checksum differs from the public v1 archive; review dataset version first')
    output.mkdir(parents=True, exist_ok=True)
    cached = output/'source-v1.zip'
    if cached.exists() and sha256(cached.read_bytes()) != sha256(raw):
        raise ValueError('Refusing to overwrite a different cached source')
    if path.resolve() != cached.resolve() and not cached.exists():
        shutil.copyfile(path, cached)
    manifest = dict(adapter='upv-sssb-v1', source=SOURCE, archive_sha256=sha256(raw), archive_md5=ARCHIVE_MD5,
                    license='Not established from inspected record; local analysis only, do not redistribute raw exports without checking terms',
                    retrieved_date='2026-10-12', split_scope='within-campaign condition holdout, NOT independent-laboratory validation',
                    experiments={})
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        for identifier, (split, member) in EXPERIMENTS.items():
            data = archive.read(member)
            records, meta = parse_export(data)
            path_out = output/f'{identifier}.csv'
            with path_out.open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0]))
                writer.writeheader(); writer.writerows(records)
            meta.update(split=split, source_member=member, source_sha256=sha256(data),
                        canonical_sha256=sha256(path_out.read_bytes()), canonical_csv=path_out.name)
            manifest['experiments'][identifier] = meta
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--output', type=Path, default=ROOT/'data/experiments/upv')
    args=parser.parse_args()
    manifest=import_archive(args.archive, args.output)
    for name, info in manifest['experiments'].items():
        print(name, info['split'], info['samples'], 'samples;', round(info['duration_s'],3), 's')
    print('Manifest:', args.output/'manifest.json')


if __name__=='__main__':
    main()
