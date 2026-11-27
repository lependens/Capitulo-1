import json
import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path
from siar_web_descarga import SCHEMA, spans, validate, save_block, export

class DownloaderTests(unittest.TestCase):
    def setUp(self):
        self.headers=['campo'+str(i) for i in range(20)]
        for i,p in SCHEMA.items(): self.headers[i]=p+str(i)
        self.row=['']*20; self.row[0]='01/01/2020'; self.row[1]='12,34'
        self.start=date(2020,1,1); self.end=date(2020,1,2)

    def test_split_leap_year(self):
        self.assertEqual(list(spans(date(2020,2,28),date(2020,3,2),2)),[(date(2020,2,28),date(2020,2,29)),(date(2020,3,1),date(2020,3,2))])

    def test_partial_result_is_pending(self):
        _,missing=validate(self.headers,[self.row],self.start,self.end)
        self.assertEqual(missing,['2020-01-02'])
        with self.assertRaises(ValueError): validate(self.headers,[],self.start,self.end)

    def test_reject_wrong_or_duplicate_dates(self):
        with self.assertRaises(ValueError): validate(self.headers,[self.row,self.row],self.start,self.end)
        row=self.row.copy(); row[0]='03/01/2020'
        with self.assertRaises(ValueError): validate(self.headers,[row],self.start,self.end)
        with self.assertRaises(ValueError): validate(['mal']*20,[self.row],self.start,self.end)

    def test_preserves_raw_values_and_missing_dates(self):
        with tempfile.TemporaryDirectory() as folder:
            db=sqlite3.connect(':memory:')
            db.execute('CREATE TABLE blocks (station TEXT,start TEXT,end TEXT,captured TEXT,sha256 TEXT,data TEXT,missing TEXT,PRIMARY KEY(station,start,end))')
            self.assertEqual(save_block(db,'AL10',self.start,self.end,self.headers,[self.row]),(1,1))
            self.assertEqual(json.loads(db.execute('select missing from blocks').fetchone()[0]),['2020-01-02'])
            export(db,Path(folder),'AL10')
            self.assertIn('12,34',(Path(folder)/'AL10_web.csv').read_text(encoding='utf-8'))

if __name__=='__main__': unittest.main()

