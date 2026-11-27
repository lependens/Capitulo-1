import tempfile,unittest
from datetime import date,timedelta
from pathlib import Path
from andalucia_compare_web import compare

HEAD='FECHA DIA '+' '.join('Al05'+s for s in 'TMax HTMax TMin HTMin TMed HumMax HumMin HumMed VelViento DirViento Rad Precip ETo'.split())+'\n----\n'
class Comparison(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.path=Path(self.tmp.name)/'input.txt'; self.days=[date(2020,1,1)+timedelta(days=i) for i in range(10)]
        lines=[d.strftime('%d-%m-%y')+' '+str(d.timetuple().tm_yday)+' 10.0 12:00 2.0 03:00 6.0 90.0 40.0 60.0 2.0 150.0 20.0 0.0 4.0' for d in self.days]
        self.path.write_text(HEAD+'\n'.join(lines)+'\n',encoding='utf-8'); self.dates=[d.isoformat() for d in self.days]
        self.ref={d:['6.0','10.0','12:00','2.0','03:00','60.0','90.0','40.0','2.0','150.0','20.0','0.0','999'] for d in self.dates}
    def tearDown(self): self.tmp.cleanup()
    def test_et0_does_not_penalise(self): self.assertEqual(compare(self.path,'AL05',self.ref,self.dates)['status'],'PASS')
    def test_meteorology_mismatch_is_review(self):
        self.ref[self.dates[0]][1]='12'; self.assertEqual(compare(self.path,'AL05',self.ref,self.dates)['status'],'REVIEW')
    def test_missing_common_variable_is_incomplete(self):
        self.ref[self.dates[0]][1]=''; self.assertEqual(compare(self.path,'AL05',self.ref,self.dates)['results'][0]['status'],'INCOMPLETE')
    def test_missing_reference_date_does_not_pass(self):
        del self.ref[self.dates[0]]
        with self.assertRaises(ValueError): compare(self.path,'AL05',self.ref,self.dates)

if __name__=='__main__': unittest.main()

