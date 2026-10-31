import { describe, expect, it } from 'vitest';
import { ahcComparisonMethods, metricGroup, number, plotPoints, splitMeaning } from './data';
describe('scientific display semantics',()=>{
 it('keeps missing and zero values distinct',()=>{expect(number(null)).toBe('—');expect(number(0)).toBe('0,000');expect(plotPoints([{date:'2025-01-01',PM:null},{date:'2025-01-02',PM:0}], 'PM')).toEqual([['2025-01-01',null],['2025-01-02',0]]);});
 it('requests published group dimensions',()=>{expect(metricGroup('monthly')).toBe('month');expect(metricGroup('seasonal')).toBe('season');expect(metricGroup('yearly')).toBe('year');expect(metricGroup('overview')).toBe('global');});
 it('keeps scientific split semantics explicit',()=>{expect(splitMeaning('TRAIN')).toContain('Ajuste');expect(splitMeaning('VALIDATION')).toContain('Selección');expect(splitMeaning('TEST')).toContain('retrospective_locked');expect(splitMeaning('ALL')).toContain('sin métrica combinada');});
 it('keeps HG-Rs annual and selected identities separate',()=>{expect(ahcComparisonMethods('HG_RS_V1')).toEqual(['HG_RS_V1','HG_RS_AHC_VALIDATED','HG_RS_AHC_VALIDATED_SELECTED']);expect(ahcComparisonMethods('HS_RA_V1')).toEqual(['HS_RA_V1','HS_RA_AHC_VALIDATED']);});
});
