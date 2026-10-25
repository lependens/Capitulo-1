import { describe, expect, it } from 'vitest';
import { metricGroup, number, plotPoints } from './data';
describe('scientific display semantics',()=>{
 it('keeps missing and zero values distinct',()=>{expect(number(null)).toBe('—');expect(number(0)).toBe('0,000');expect(plotPoints([{date:'2025-01-01',PM:null},{date:'2025-01-02',PM:0}], 'PM')).toEqual([['2025-01-01',null],['2025-01-02',0]]);});
 it('requests published group dimensions',()=>{expect(metricGroup('monthly')).toBe('month');expect(metricGroup('seasonal')).toBe('season');expect(metricGroup('yearly')).toBe('year');expect(metricGroup('overview')).toBe('global');});
});
