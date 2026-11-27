import hashlib,json,sqlite3,tempfile,unittest
from datetime import date,timedelta
from pathlib import Path
from andalucia_web_runner import audit_one,setup,validate_plan

SUFFIXES='TMax HTMax TMin HTMin TMed HumMax HumMin HumMed VelViento DirViento Rad Precip ETo'.split()
class RunnerSafety(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name); self.source=self.root/'data.txt'
        days=[date(2020,1,1)+timedelta(days=i) for i in range(10)]
        head='FECHA DIA '+' '.join('Al05'+x for x in SUFFIXES)+'\n---\n'
        self.source.write_text(head+'\n'.join(d.strftime('%d-%m-%y')+' '+str(d.timetuple().tm_yday)+' 10.0 12:00 2.0 03:00 6.0 90.0 40.0 60.0 2.0 150.0 20.0 0.0 4.0' for d in days)+'\n',encoding='utf8')
        self.entry={'station_candidate':'AL05','file':'data.txt','sha256':hashlib.sha256(self.source.read_bytes()).hexdigest(),'windows':[{'start':'2020-01-01','end':'2020-01-05'},{'start':'2020-01-06','end':'2020-01-10'}]}
        self.db=sqlite3.connect(':memory:');setup(self.db);self.calls=[]
    def tearDown(self): self.db.close();self.tmp.cleanup()
    def fetch(self,st,start,end):
        self.calls.append(start);d=date.fromisoformat(start)
        return {(d+timedelta(days=i)).isoformat():['6.0','10.0','12:00','2.0','03:00','60.0','90.0','40.0','2.0','150.0','20.0','0.0','999'] for i in range(5)}
    def test_interruption_resumes_without_repeating_complete_block(self):
        def interrupted(st,start,end):
            if start=='2020-01-06': raise OSError('offline')
            return self.fetch(st,start,end)
        with self.assertRaises(OSError): audit_one(self.entry,self.root,'AL05',self.db,interrupted,0)
        self.assertEqual(self.db.execute('SELECT COUNT(*) FROM blocks').fetchone()[0],1)
        result=audit_one(self.entry,self.root,'AL05',self.db,self.fetch,0)
        self.assertEqual(result['status'],'PASS');self.assertEqual(self.calls,['2020-01-01','2020-01-06'])
    def test_missing_day_never_saves_complete_block(self):
        with self.assertRaises(ValueError): audit_one(self.entry,self.root,'AL05',self.db,lambda *args:{},0)
        self.assertEqual(self.db.execute('SELECT COUNT(*) FROM blocks').fetchone()[0],0)
    def test_source_change_and_traversal_rejected(self):
        self.source.write_text('changed',encoding='utf8')
        with self.assertRaises(ValueError): validate_plan(self.entry,self.root,'AL05')
        self.entry['file']='../data.txt'
        with self.assertRaises(ValueError): validate_plan(self.entry,self.root,'AL05')
if __name__=='__main__': unittest.main()

