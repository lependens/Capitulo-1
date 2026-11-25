import importlib.util,tempfile
from pathlib import Path
s=importlib.util.spec_from_file_location('r',str(Path(__file__).resolve().parents[1]/'ops/reconcile.py'));r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
h=b'Fecha;Kc;ET0;ETc;Pe;ETc - Pe\n'
with tempfile.TemporaryDirectory() as d:
 p=Path(d)/'forecast.csv';p.write_bytes(h);r.validate_csv_and_zip(Path(d))
 for data in (b'',b'bad\n',h+b'25/11/2026;0,8\n',h+b'bad;0,8;2;2;0;2\n'):
  p.write_bytes(data)
  try:r.validate_csv_and_zip(Path(d))
  except r.ReconcileError:continue
  raise AssertionError(data)
print('CSV validation passed: header-only accepted; malformed rejected')
