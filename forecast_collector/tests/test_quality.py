from siar_forecast_raw.quality import classify_csv
H = b'Fecha;Kc;ET0;ETc;Pe;ETc - Pe\n'
def test_empty(): assert classify_csv(H)['status'] == 'empty'
def test_useful(): assert classify_csv(H+b'25/11/2026;0,8;2,5;2;0;2\n')['status'] == 'useful'
def test_partial(): assert classify_csv(H+b'25/11/2026;0,8;2,5;2;0;2\n26/11/2026;0,8;;;;\n')['status'] == 'partial'
def test_missing_values(): assert classify_csv(H+b'25/11/2026;0,8;;;;\n')['status'] == 'partial'
def test_invalid(): assert classify_csv(b'bad\n1\n')['status'] == 'failed'
def test_nonfinite(): assert classify_csv(H+b'25/11/2026;0,8;NaN;;;\n')['status'] == 'partial'
