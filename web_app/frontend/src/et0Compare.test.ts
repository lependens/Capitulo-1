import {describe,expect,it} from 'vitest';
import {csvCell,difference} from './et0Compare';
import type {Result} from './et0Compare';
const reference:Result={method:'AHC',calendar:'annual',key:'IB01',n:2192,cohort_id:'abc',rmse:0.5,
  mae:0.4,bias:-0.1,r2:0.8,coverage_calendar:1,calendar_days:2192,predicted_days:2192};
describe('comparaciones retrospectivas',()=>{
  it('autoriza una mejora entre calendarios con claves idénticas',()=>{
    const comparison=difference({...reference,calendar:'monthly',rmse:0.375},reference,'rmse');
    expect(comparison).toEqual({delta:-0.125,improvement:25});
  });
  it('bloquea diferencias con distinta cohorte, soporte o grupo',()=>{
    for(const overrides of [{cohort_id:'xyz'},{n:100},{key:'IB02'},{cohort_id:undefined},{rmse:null}]){
      expect(difference({...reference,...overrides},reference,'rmse')).toBeNull();
    }
  });
  it('mantiene el signo del sesgo y evita una mejora porcentual indefinida',()=>{
    expect(difference({...reference,bias:0.1},reference,'bias')).toEqual({delta:0.2,improvement:null});
    expect(difference(reference,{...reference,mae:0},'mae')?.improvement).toBeNull();
    expect(difference(reference,reference,'coverage_calendar')).toBeNull();
  });
  it('escapa comillas en CSV',()=>expect(csvCell('n,"PM"')).toBe('"n,""PM"""'));
});
