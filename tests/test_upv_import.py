import importlib.util
import io
from pathlib import Path
import unittest
import zipfile

spec=importlib.util.spec_from_file_location('upv_import', Path(__file__).resolve().parents[1]/'scripts/import_upv.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def book(rows):
    xml='<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
    for i, values in enumerate(rows, 1):
        xml+=f'<row r="{i}">'
        for col,value in values.items():
            if isinstance(value,str):
                xml+=f'<c r="{col}{i}" t="inlineStr"><is><t>{value}</t></is></c>'
            else:
                xml+=f'<c r="{col}{i}"><v>{value}</v></c>'
        xml+='</row>'
    xml+='</sheetData></worksheet>'
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w') as z:z.writestr('xl/worksheets/sheet1.xml',xml)
    return stream.getvalue()


class UPVImportTests(unittest.TestCase):
    def rows(self):
        return [{'A':'Scan Sweep Time (Sec)','B':'Scan Number','C':'101 (°C)','E':'114 (Vdc)'},
                {'A':45000,'B':1,'C':25,'E':.4},
                {'A':45000+1/86400,'B':2,'C':9.9e37,'E':.5}]

    def test_time_units_sparse_columns_and_quality(self):
        rows,meta=module.parse_export(book(self.rows()))
        self.assertAlmostEqual(rows[1]['time_s'],1,places=5)
        self.assertEqual(rows[0]['T101_k'],298.15)
        self.assertEqual(rows[1]['T101_k'],'')
        self.assertEqual(rows[1]['q101'],'instrument_overrange')
        self.assertEqual(rows[1]['U114_v'],.5)
        self.assertEqual(meta['channel_quality_counts']['101']['valid'],1)

    def test_duplicate_scan_rejected(self):
        rows=self.rows();rows[2]['B']=1
        with self.assertRaisesRegex(ValueError,'scan'):module.parse_export(book(rows))

    def test_seconds_not_silently_multiplied_by_86400(self):
        rows=self.rows();rows[1]['A']=0
        with self.assertRaisesRegex(ValueError,'Excel'):module.parse_export(book(rows))

    def test_missing_data_header_rejected(self):
        with self.assertRaisesRegex(ValueError,'header'):module.parse_export(book([{'A':'wrong'}]))


if __name__=='__main__':unittest.main()
