import tempfile
import unittest
from pathlib import Path
from andalucia_stage import parse_file

SUFFIXES = 'TMax HTMax TMin HTMin TMed HumMax HumMin HumMed VelViento DirViento Rad Precip ETo'.split()
def header(station, suffixes=SUFFIXES): return 'FECHA DIA '+ ' '.join(station+x for x in suffixes)+'\n-------- ---\n'
HEADER = header('Al05')
VALUES = ' 1 12:00 2 03:00 3 90 40 60 2 150 20 0 4'

class ImportSafety(unittest.TestCase):
    def parse(self, text):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'data.txt'; p.write_text(HEADER+text, encoding='utf-8')
            return parse_file(p,'AL05',2000)

    def test_date_day_disagreement_blocks_both_candidates(self):
        result=self.parse('30-12-15 365'+VALUES+'\n29-12-15 364'+VALUES+'\n29-12-15 363'+VALUES+'\n')
        self.assertEqual(result['rows'],[])
        self.assertEqual([e['reason'] for e in result['exceptions']].count('date_day_mismatch'),2)
        self.assertEqual([e['reason'] for e in result['exceptions']].count('date_has_ambiguous_evidence'),1)

    def test_date_only_is_unresolved_and_valid_row_retains_et0(self):
        result=self.parse('01-06-20\n02-06-20 154'+VALUES+'\n')
        self.assertEqual(result['exceptions'][0]['reason'],'date_only')
        self.assertEqual(result['rows'][0]['EtPMon'],'4')
        self.assertEqual(result['rows'][0]['TempSuelo1'],'')

    def test_duplicate_is_not_imported(self):
        result=self.parse(('01-06-20 153'+VALUES+'\n')*2)
        self.assertEqual(result['rows'],[])
        self.assertEqual(len(result['exceptions']),2)

    def test_nonfinite_or_ambiguous_times_are_not_imported(self):
        result=self.parse('01-06-20 153 1 24:00 2 03:00 3 90 40 60 2 150 20 0 4\n')
        self.assertEqual(result['exceptions'][0]['reason'],'invalid_value')

    def test_date_after_known_source_period_is_quarantined(self):
        from datetime import date
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'data.txt'; p.write_text(HEADER+'01-01-99 1'+VALUES+'\n',encoding='utf-8')
            result=parse_file(p,'AL05',2000,max_date=date(2020,6,5))
            self.assertEqual(result['exceptions'][0]['reason'],'date_outside_source_period')

    def test_proposed_alias_requires_explicit_header(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'data.txt'; p.write_text(header('Hu10')+'01-06-20 153'+VALUES+'\n',encoding='utf-8')
            with self.assertRaises(ValueError): parse_file(p,'H10',2000)
            self.assertEqual(len(parse_file(p,'H10',2000,header_label='HU10')['rows']),1)

    def test_radiation_only_header_preserves_missing_fields(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'data.txt'; p.write_text(header('Hu103',SUFFIXES[:11])+'01-06-20 153 1 12:00 2 03:00 3 90 40 60 2 150 20\n',encoding='utf-8')
            row=parse_file(p,'H103',2000,header_label='HU103')['rows'][0]
            self.assertEqual((row['Radiacion'],row['Precipitacion'],row['EtPMon']),('20','',''))

if __name__=='__main__': unittest.main()

