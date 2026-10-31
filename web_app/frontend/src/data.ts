export interface Model {id:string;label:string;color:string;reference?:boolean;inputs?:string|string[];base_method_id?:string;granularity?:string;adjustment_kind?:string;selected?:boolean}
export interface Station {code:string;name:string;status:string;from:string;to:string;rows:number;quality_status:string}
export interface Metric {model:string;cohort:string;group:string;key:string;n:number;mae:number|null;rmse:number|null;bias:number|null;r2:number|null}
export interface Series {points:({date:string}&Record<string,string|number|null>)[];aggregation:string;source_rows:number;returned_points:number}
export interface Scatter {points:{reference:number;predicted:number;date:string}[];total_rows:number;returned_points:number;sampling:string}
export interface Coverage {code:string;rows:number;quality_status:string;models:{model:string;predicted:number;evaluated:number}[]}
export interface AhcMetric {split:string;reference_id:string;group:string;key:string;station_code:string|null;cohort_id:string;method_id:string;n:number;mae:number|null;rmse:number|null;bias:number|null;r2:number|null;r2_status:string}
export interface AhcCoefficient {coefficient_id:string;station_code:string;base_method_id:string;kind:string;reference_id:string;granularity:string;group_key:string;coefficient:number|null;status:string;deployable:boolean;fit_split:string;calibration_start:string|null;calibration_end:string|null;support:Record<string,unknown>;estimate:Record<string,unknown>}
export interface AhcSeries extends Series {split:string;data_recalculated:boolean}
export type ScientificSplit='ALL'|'TRAIN'|'VALIDATION'|'TEST';
export function number(value:number|null|undefined,digits=3){return value==null?'—':value.toLocaleString('es-ES',{minimumFractionDigits:digits,maximumFractionDigits:digits});}
export async function api<T>(path:string,signal?:AbortSignal):Promise<T>{const response=await fetch(`/api/${path}`,{signal});if(!response.ok){let detail=`Error ${response.status}`;try{const body=await response.json();detail=typeof body.detail==='string'?body.detail:detail;}catch{/* HTTP status remains useful. */}throw new Error(detail);}return response.json() as Promise<T>;}
export function metricGroup(tab:string){return ({monthly:'month',seasonal:'season',yearly:'year'} as Record<string,string>)[tab]??'global';}
export function plotPoints(points:Series['points'],model:string){return points.map(p=>[p.date,p[model]??null]);}
export function splitMeaning(split:ScientificSplit){return ({ALL:'Filas publicadas de TRAIN, VALIDATION y TEST; sin métrica combinada.',TRAIN:'Ajuste de coeficientes AHC.',VALIDATION:'Selección de granularidad congelada antes de TEST.',TEST:'Evaluación retrospective_locked; no es confirmación prospectiva.'} as Record<ScientificSplit,string>)[split];}
export function ahcComparisonMethods(base:string){return base==='HG_RS_V1'?[base,'HG_RS_AHC_VALIDATED','HG_RS_AHC_VALIDATED_SELECTED']:[base,base.replace('_V1','_AHC_VALIDATED')];}
