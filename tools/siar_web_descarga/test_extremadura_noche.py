import unittest
from extremadura_noche import normalize,COLUMNS

class ConversionTests(unittest.TestCase):
    def setUp(self):
        self.h=['Fecha','Temp Media (ºC)','Temp Max (ºC)','Hora Tª Max','Temp Mín (ºC)','Hora Tª Min','Hum Media (%)','Hum Max (%)','Hora Hum Max','Hum Min (%)','Hora Hum Mín','Vel Viento (m/s)','Dir Viento (º)','Vel V. Max (m/s)','Hora Vel Max V.','Dir V. Vel Max (º)','Rad (MJ/m2)','Precip (mm)','P. Efect (mm)','ET0 (mm)']
        self.row=['14/03/2012','14,71','20,64','09:30','9,40','00:00','57,91','84,20','19:20','35,79','08:20','0,97','5,24','4,41','05:23','0,00','21,76','0,00','0,10','3,08']
    def test_canonical_mapping(self):
        r=dict(zip(COLUMNS,normalize('BA01',self.h,self.row)))
        self.assertEqual(len(r),23);self.assertEqual(r['Fecha'],'2012-03-14');self.assertEqual(r['HorMinVelMax'],'523');self.assertEqual(r['EtPMon'],'3.08');self.assertEqual(r['PePMon'],'0.10');self.assertEqual(r['TempSuelo1'],'');self.assertEqual(r['Estacion'],'BA01')
    def test_schema_and_nonfinite_rejected(self):
        h=self.h.copy();h[18]='otra cosa'
        with self.assertRaises(ValueError): normalize('BA01',h,self.row)
        row=self.row.copy();row[1]='NaN'
        with self.assertRaises(ValueError): normalize('BA01',self.h,row)
if __name__=='__main__': unittest.main()

