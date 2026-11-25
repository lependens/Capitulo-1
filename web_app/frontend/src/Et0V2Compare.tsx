import { lazy, Suspense, useState } from 'react';
import { Select, useRead } from './Et0V2Preview';
import type { Catalog } from './Et0V2Preview';
import { calendarNames, csvCell, difference, metricNames, rowId } from './et0Compare';
import type { Metric, Result } from './et0Compare';

const Chart = lazy(() => import('./Chart'));
const metrics = Object.keys(metricNames) as Metric[];
const number = new Intl.NumberFormat('es-ES',{maximumFractionDigits:5});
export const fmt = (v:number|null|undefined) => v == null ? '—' : number.format(v);
const names:Record<string,string> = {
  FORMULA:'Fórmula original',AHC:'AHC',ML:'ML principal',ANN_A:'ANN A',ANN_E:'ANN E',
  ML_LINEAR:'Linear Regression',ML_RIDGE:'Ridge',ML_ELASTIC_NET:'Elastic Net',ML_TREE:'Decision Tree',
  ML_RANDOM_FOREST:'Random Forest',ML_EXTRA_TREES:'Extra Trees',ML_HIST_GRADIENT_BOOSTING:'Hist Gradient Boosting',
};
export const methodName = (id:string) => names[id] ?? id;
export const methodColor = (id:string) => ({FORMULA:'#64748b',AHC:'#a65308',ANN_A:'#7151b1',ANN_E:'#a33061',reference:'#253546'}[id] ?? '#087768');
const list = (options:Record<string,string>, value:string[], onChange:(v:string[])=>void) =>
  Object.entries(options).map(([id,label])=><label key={id}><input type="checkbox" checked={value.includes(id)}
    onChange={e=>onChange(e.target.checked?[...value,id]:value.filter(v=>v!==id))}/>{label}</label>);

export type Preset = { methods?:string[]; calendars?:string[]; station?:string; allStations?:boolean };
export function Compare({catalog,preset={}}:{catalog:Catalog;preset?:Preset}) {
  const [formula,setFormula] = useState('HG_RS_V1');
  const [methods,setMethods] = useState(preset.methods ?? ['FORMULA','AHC','ML','ANN_A','ANN_E']);
  const [calendars,setCalendars] = useState(preset.calendars ?? ['annual','seasonal','monthly']);
  const stationOptions = Object.fromEntries((catalog.group_keys.station ?? []).filter(k=>catalog.station_details?.some(s=>s.code===k&&s.retrospective)).map(k=>[k,catalog.stations[k]??k]));
  const [stations,setStations] = useState(preset.allStations ? Object.keys(stationOptions) : preset.station ? [preset.station] : []);
  const [year,setYear] = useState('');
  const [metric,setMetric] = useState<Metric>('rmse');
  const [pivot,setPivot] = useState('rows');
  const [sort,setSort] = useState<keyof Result>('calendar');
  const [reverse,setReverse] = useState(false);
  const [baseline,setBaseline] = useState('');
  const group = stations.length?(year?'station_year':'station'):(year?'year':'global');
  const keys = stations.length?stations.map(k=>year?k+':'+year:k):[year||'all'];
  const query = new URLSearchParams({formula,group});
  calendars.forEach(v=>query.append('calendar',v));
  methods.forEach(v=>query.append('method',v));
  keys.forEach(v=>query.append('key',v));
  const enabled = calendars.length>0 && methods.length>0;
  const response = useRead<{rows:Result[]}>(enabled?'comparison?'+query:'');
  const rows = [...(response?.data?.rows ?? [])].sort((a,b)=>{
    const x=a[sort], y=b[sort];
    if(x==null) return y==null?0:1;
    if(y==null) return -1;
    return (typeof x==='number'&&typeof y==='number'?x-y:String(x).localeCompare(String(y)))*(reverse?-1:1);
  });
  const label=(r:Result) => methodName(r.method)+' · '+calendarNames[r.calendar];
  const seriesKeys=[...new Set(rows.map(r=>r.method+'/'+r.calendar))];
  const delta=(r:Result)=>difference(r,rows.find(candidate=>candidate.key===r.key&&candidate.method+'/'+candidate.calendar===baseline),metric);
  const shown = (r:Result|undefined,m:Metric) => fmt(m==='coverage_calendar'&&r?.[m]!=null?r[m]*100:r?.[m]);
  function reset(){
    setFormula('HG_RS_V1');setMethods(preset.methods??['FORMULA','AHC','ML','ANN_A','ANN_E']);
    setCalendars(preset.calendars??['annual','seasonal','monthly']);
    setStations(preset.allStations?Object.keys(stationOptions):preset.station?[preset.station]:[]);
    setYear('');setMetric('rmse');setBaseline('');setPivot('rows');setSort('calendar');setReverse(false);
  }
  function exportRows(){
    const visible = pivot==='rows' ?
      [['Estación o grupo','Método','Temporalidad','n','RMSE','MAE','Sesgo','R²','Cobertura (%)','Días previstos','Días calendario','SHA cohorte','Diferencia '+metric,'% mejora'],
       ...rows.map(r=>[r.key,r.method,r.calendar,r.n,r.rmse,r.mae,r.bias,r.r2,r.coverage_calendar==null?null:r.coverage_calendar*100,r.predicted_days,r.calendar_days,r.cohort_id,delta(r)?.delta,delta(r)?.improvement])] :
      [[pivot==='columns'?'Estación o grupo':'Método y temporalidad',...(pivot==='columns'?seriesKeys:keys)],
        ...(pivot==='columns'?keys:seriesKeys).map(id=>[id,...(pivot==='columns'?seriesKeys:keys).map(other=>{
          const r=rows.find(r=>r.key===(pivot==='columns'?id:other)&&r.method+'/'+r.calendar===(pivot==='columns'?other:id));
          return metric==='coverage_calendar'&&r?.[metric]!=null?r[metric]*100:r?.[metric];
        })])];
    const url=URL.createObjectURL(new Blob(['\ufeff'+visible.map(row=>row.map(csvCell).join(',')).join('\n')],{type:'text/csv;charset=utf-8'}));
    const a=document.createElement('a');a.href=url;a.download='comparacion-et0-v2.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
  const column = (key:keyof Result,title:string) => <th scope="col" aria-sort={sort===key?(reverse?'descending':'ascending'):'none'}>
    <button onClick={()=>{setReverse(sort===key?!reverse:false);setSort(key);}}>{title}</button></th>;
  return <>
    <section className="p2-panel">
      <div className="p2-filters">
        <Select label="Familia de entradas" value={formula} onChange={setFormula} options={Object.fromEntries(Object.entries(catalog.formulas).map(([k,v])=>[k,v.label]))}/>
        <Select label="Periodo publicado" value={year} onChange={setYear} options={{'':'2020–2025',...Object.fromEntries((catalog.group_keys.year??[]).map(y=>[y,y]))}}/>
        <Select label="Métrica del gráfico" value={metric} onChange={v=>setMetric(v as Metric)} options={metricNames}/>
      </div>
      <p>{catalog.formulas[formula]?.inputs.join(' · ')}</p>
      <fieldset><legend>Configuración del modelo</legend>{list(calendarNames,calendars,setCalendars)}</fieldset>
      <details><summary>Estaciones · {stations.length ? stations.join(', ') : 'Agregado de las nueve estaciones'}</summary>
        <p>Sin selección se usa el agregado publicado. Seleccionar varias estaciones muestra sus filas separadas.</p>
        <button onClick={()=>setStations(Object.keys(stationOptions))}>Todas por separado</button>{' '}
        <button onClick={()=>setStations([])}>Ver agregado</button>
        <fieldset><legend>Estaciones con retrospectiva</legend>{list(stationOptions,stations,setStations)}</fieldset>
      </details>
      <details open={preset.methods?.length===7||undefined}><summary>Métodos · {methods.length} elegidos</summary>
        <fieldset><legend>Métodos y variantes</legend>{list(Object.fromEntries(Object.keys(catalog.methods).map(id=>[id,methodName(id)])),methods,setMethods)}</fieldset>
      </details>
      <button onClick={reset}>Restablecer filtros</button>
    </section>
    {!enabled?<p role="status">Selecciona al menos un método y una configuración.</p>:!response?.data?
      <p role="status">{response?.error??'Consultando resultados…'}</p>:rows.length===0?
      <p role="status">El catálogo no contiene filas para los grupos seleccionados.</p>:
      <section className="p2-panel">
        <h2>Resultados retrospectivos</h2>
        <p>Fuente: retrospective-metrics.csv · {catalog.version} · {catalog.reference} · {year||'2020–2025'}.</p>
        <Suspense fallback={<p>Cargando gráfico…</p>}><Chart label={'Comparación · '+metricNames[metric]} option={{
          tooltip:{trigger:'axis'},legend:{type:'scroll',bottom:0},grid:{left:62,right:24,bottom:52},
          xAxis:{type:'category',data:keys},yAxis:{type:'value',name:metricNames[metric]},animation:false,
          series:seriesKeys.map(id=>{const example=rows.find(r=>r.method+'/'+r.calendar===id)!;return {
            name:label(example),type:'bar',itemStyle:{color:methodColor(example.method)},data:keys.map(k=>{
              const row=rows.find(r=>r.key===k&&r.method+'/'+r.calendar===id);
              return row?.[metric]==null?null:row[metric]*(metric==='coverage_calendar'?100:1);
            })};}),
        }}/></Suspense>
        <div className="p2-filters">
          <Select label="Disposición de tabla" value={pivot} onChange={setPivot} options={{rows:'Detalle de métricas',columns:'Estaciones en filas',transpose:'Métodos en filas'}}/>
          <Select label="Comparar contra" value={baseline} onChange={setBaseline} options={{'':'Sin referencia de comparación',...Object.fromEntries(seriesKeys.map(id=>[id,label(rows.find(r=>r.method+'/'+r.calendar===id)!)]))}}/>
          <button onClick={exportRows}>Exportar CSV</button>
        </div>
        <details><summary>Interpretar la comparación</summary><p>Las diferencias se muestran si el grupo, la huella de fechas y n coinciden. La mejora relativa se calcula para RMSE y MAE con referencia positiva. Sesgo es estimación menos PM. Cobertura es disponibilidad y tiene su propio denominador. Un mínimo observado en TEST es descriptivo.</p></details>
        <div className="v2-table" tabIndex={0} role="region" aria-label="Resultados y cohortes">
          {pivot==='rows'?<table><thead><tr>{column('key','Grupo')}{column('method','Método')}{column('calendar','Configuración')}{column('n','Pares')}
            {metrics.map(m=><Column key={m}>{column(m,metricNames[m])}</Column>)}<th>Denominador</th><th>Cohorte SHA</th>{baseline&&<><th>Δ métrica</th><th>Mejora</th></>}</tr></thead>
            <tbody>{rows.map(r=><tr key={rowId(r)}><th scope="row">{r.key}</th><td>{methodName(r.method)}</td><td>{calendarNames[r.calendar]}</td><td>{r.n}</td>
              {metrics.map(m=><td key={m}>{shown(r,m)}{m==='coverage_calendar'&&r[m]!=null?' %':''}</td>)}
              <td>{r.predicted_days} / {r.calendar_days}</td><td><abbr title={r.cohort_id}>{r.cohort_id?.slice(0,10)??'—'}</abbr></td>
              {baseline&&<><td>{fmt(delta(r)?.delta)}</td><td>{delta(r)?.improvement!=null?fmt(delta(r)?.improvement)+' %':'—'}</td></>}</tr>)}</tbody></table>:
            <table><caption>{metricNames[metric]}{metric==='coverage_calendar'?' (%)':''}</caption>
              <thead><tr><th>{pivot==='columns'?'Grupo':'Método · configuración'}</th>{(pivot==='columns'?seriesKeys:keys).map(id=><th key={id}>{pivot==='columns'?label(rows.find(r=>r.method+'/'+r.calendar===id)!):id}</th>)}</tr></thead>
              <tbody>{(pivot==='columns'?keys:seriesKeys).map(id=><tr key={id}><th scope="row">{pivot==='columns'?id:label(rows.find(r=>r.method+'/'+r.calendar===id)!)}</th>{(pivot==='columns'?seriesKeys:keys).map(other=><td key={other}>{shown(rows.find(r=>r.key===(pivot==='columns'?id:other)&&r.method+'/'+r.calendar===(pivot==='columns'?other:id)),metric)}</td>)}</tr>)}</tbody></table>}
        </div>
      </section>}
  </>;
}
function Column({children}:{children:React.ReactNode}) { return <>{children}</>; }
