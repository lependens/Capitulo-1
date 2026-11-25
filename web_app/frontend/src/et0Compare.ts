export type Result = {
  family?:string; method:string; calendar:string; key:string; n:number; cohort_id?:string;
  rmse:number|null; mae:number|null; bias:number|null; r2:number|null;
  coverage_calendar:number|null; calendar_days:number; predicted_days:number;
};
export const calendarNames: Record<string,string> = {annual:'Anual',seasonal:'Estacional',monthly:'Mensual'};
export const metricNames = {rmse:'RMSE · mm/día',mae:'MAE · mm/día',bias:'Sesgo · mm/día',r2:'R²',coverage_calendar:'Cobertura'};
export type Metric = keyof typeof metricNames;
export const rowId = (row: Result) => [row.key,row.family,row.method,row.calendar].join('/');
export function difference(row: Result, reference: Result | undefined, metric: Metric) {
  if (!reference || row.key !== reference.key || !row.cohort_id ||
      row.cohort_id !== reference.cohort_id || row.n <= 0 || row.n !== reference.n ||
      row[metric] == null || reference[metric] == null || metric === 'coverage_calendar') return null;
  const delta = row[metric] - reference[metric];
  const improvesMagnitude = metric === 'rmse' || metric === 'mae';
  return {delta, improvement: improvesMagnitude && reference[metric] > 0 ? -100 * delta / reference[metric] : null};
}
export function csvCell(value: unknown) {
  return '"' + String(value ?? '').replaceAll('"','""') + '"';
}
