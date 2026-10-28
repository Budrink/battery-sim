"""Portable, lossless CSV export of the exact recorded comparison inputs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import zipfile
import numpy as np


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def write_csv(path,columns,rows):
    with path.open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(columns);writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();run=args.run.resolve();out=args.output.resolve()
    if out.exists() or out.with_suffix('.zip').exists():raise FileExistsError(out)
    m=json.loads((run/'visual/manifest.json').read_text())
    if m['partial']:raise ValueError('Incomplete recording')
    scene=json.loads((run/'scenario.json').read_text())
    ids=sorted(c['id'] for c in scene['cells'] if c.get('role')=='battery')
    t=np.asarray(m['time_s'],dtype=float)
    width=4*len(m['objects'])+m['channel_fields']*len(m['channels'])+m['gas_fields']*len(m['gas'])
    frames_path=run/'visual/frames.bin'
    if frames_path.stat().st_size!=len(t)*width*4:raise ValueError('Recording length mismatch')
    frames=np.memmap(frames_path,dtype='<f4',mode='r',shape=(len(t),width))
    model=np.asarray(frames[:,[4*m['objects'].index(i) for i in ids]],dtype=float)-273.15
    exp=json.loads((run/'experiment.json').read_text())
    et=np.asarray(exp['time_s'],dtype=float)
    keep=(et>=0)&(et<=t[-1]+1e-6);et=et[keep]
    sensors=[f'T{i}' for i in range(1,10)]
    measured=np.stack([np.asarray(exp[s],dtype=float)[keep] for s in sensors],axis=1)
    tolerance=.05*(measured+273.15)
    low,high=measured-tolerance,measured+tolerance
    assert np.isfinite(model).all() and np.isfinite(measured).all()
    assert np.all(np.diff(t)>0) and np.all(np.diff(et)>0)
    out.mkdir(parents=True)
    write_csv(out/'model_temperatures.csv',['time_s']+[f'{i}_temperature_c' for i in ids],np.column_stack((t,model)))
    columns=['time_s'];blocks=[et[:,None]]
    for j,s in enumerate(sensors):
        columns.extend([f'{s}_temperature_c',f'{s}_lower_c',f'{s}_upper_c'])
        blocks.extend([measured[:,j,None],low[:,j,None],high[:,j,None]])
    experimental=np.column_stack(blocks)
    write_csv(out/'experiment_with_tolerance.csv',columns,experimental)
    # Exact displayed series in tidy form, preserving each original time axis.
    rows=[]
    for panel,names,sensor in [('primary',['B10'],'T5'),('secondary',['B17','B18'],'T4')]:
        for name in names:
            j=ids.index(name)
            rows.extend((panel,'model',name,float(x),float(y),'','') for x,y in zip(t,model[:,j]))
        j=sensors.index(sensor)
        rows.extend((panel,'experiment',sensor,float(x),float(y),float(a),float(b))
            for x,y,a,b in zip(et,measured[:,j],low[:,j],high[:,j]))
    write_csv(out/'plotted_series_long.csv',
        ['panel','source','series_id','time_s','temperature_c','lower_c','upper_c'],rows)
    # Re-read CSVs: fail if export changes any numeric values.
    np.testing.assert_array_equal(np.loadtxt(out/'model_temperatures.csv',delimiter=',',skiprows=1),np.column_stack((t,model)))
    np.testing.assert_array_equal(np.loadtxt(out/'experiment_with_tolerance.csv',delimiter=',',skiprows=1),experimental)
    with (out/'plotted_series_long.csv').open(encoding='utf-8',newline='') as f:
        restored=list(csv.DictReader(f))
    assert len(restored)==len(rows)
    for original,row in zip(rows,restored):
        assert row['panel']==original[0] and row['source']==original[1] and row['series_id']==original[2]
        for col,index in [('time_s',3),('temperature_c',4),('lower_c',5),('upper_c',6)]:
            assert row[col]=='' if original[index]=='' else float(row[col])==original[index]
    readme='''BatterySim — данные для воспроизведения графиков

Источник: завершённый fsri30-wall-delay-full-20261027-r1, интервал 0–3000 с.
Это тот же прогон и те же данные, что на приложенном comparison.png.

CSV: UTF-8, разделитель запятая, десятичная точка, первая строка — заголовки.
Время time_s — секунды от начала опыта. Все температуры *_c — градусы Цельсия.

1. plotted_series_long.csv — непосредственно линии двух панелей графика.
   panel: primary / secondary. source: model / experiment.
   series_id: B10, B17, B18 либо T5, T4.
   lower_c / upper_c: границы полосы для эксперимента; у модели пустые.
   Строки сгруппированы по линии, время упорядочено внутри каждой линии.

2. model_temperatures.csv — все 30 батарей B00–B29, wide dataframe.
   Температуры средние по батарее, из float32 записи 3D-визуализатора.
   Преобразование в Цельсии выполнено после чтения float32 в float64.
   Это не максимальная температура и не температура термопары.

3. experiment_with_tolerance.csv — все 9 термопар T1–T9 и полосы допуска.
   Исходные измерения в пределах показанного интервала, без сглаживания.

Допуск: lower_c = temperature_c - 0.05*(temperature_c + 273.15);
        upper_c = temperature_c + 0.05*(temperature_c + 273.15).
Это условный инженерный допуск ±5% абсолютной температуры в кельвинах,
НЕ доверительный интервал измерений и НЕ подтверждённая точность модели.

Как воспроизвести график:
  верхняя панель: эксперимент T5 с полосой + модель B10;
  нижняя: эксперимент T4 с полосой + модель B17 и B18.
  Сопоставление T4 ↔ B17/B18 — гипотеза, не установленная привязка датчика.
  Цвета: эксперимент #b66620, полоса #e9ac61 (alpha 0.30),
         B10/B17 #087f8c, B18 #7058a3 (пунктир).

Временные сетки РАЗНЫЕ: эксперимент 1 с; модель сохранена с переменным
интервалом, чаще в аварийной фазе. Они намеренно НЕ приведены к одной сетке.
Используйте свой time_s у каждой линии. Не соединяйте строки по их номеру.
Последнее расчётное время близко к 3000 с с обычной floating-point погрешностью.
Временного сдвига, нормировки амплитуды и сглаживания в экспорте нет.

Пример Python/pandas:
  import pandas as pd
  data = pd.read_csv('plotted_series_long.csv')
  for (panel, source, name), group in data.groupby(['panel','source','series_id']):
      x = group.time_s
      y = group.temperature_c
      # ax.plot(x, y, label=name)
      # для source == 'experiment': ax.fill_between(x, group.lower_c, group.upper_c)

metadata.json содержит хеши источников, размерности и результаты проверки CSV.
'''
    (out/'README.txt').write_text(readme,encoding='utf-8')
    for filename in ['comparison.png','comparison.pdf']:
        shutil.copy2(run/'comparison-tolerance'/filename,out/filename)
    metadata=dict(run_id=run.name,format='CSV UTF-8 comma separated decimal dot',
        time_unit='s',temperature_unit='degC',model_rows=len(t),experiment_rows=len(et),
        plotted_rows=len(rows),model_batteries=ids,experiment_sensors=sensors,
        numeric_csv_roundtrip_exact=True,smoothing=False,time_shift_s=0,interpolation=False,
        tolerance=dict(fraction_kelvin=.05,kind='assumed engineering tolerance; not measurement uncertainty'),
        mapping=dict(primary='T5 vs B10',secondary='T4 vs B17/B18: unconfirmed hypothesis'),
        source_sha256={str(p.relative_to(run)):digest(p) for p in
            [run/'scenario.json',run/'experiment.json',run/'visual/manifest.json',frames_path]},
        plot_script_sha256=digest(Path(__file__).with_name('plot_fsri30_tolerance.py')),
        exported_sha256={p.name:digest(p) for p in out.iterdir() if p.is_file()})
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    archive=out.with_suffix('.zip')
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for file in sorted(out.iterdir()):z.write(file,file.name)
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps(dict(archive=str(archive),model_rows=len(t),experiment_rows=len(et),plotted_rows=len(rows),zip_bytes=archive.stat().st_size,verified=True),indent=2))


if __name__=='__main__':main()
