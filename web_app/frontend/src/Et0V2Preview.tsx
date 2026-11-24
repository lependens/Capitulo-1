import { lazy, Suspense, useEffect, useState } from 'react';
import './et0-v2-preview.css';

const Chart = lazy(() => import('./Chart'));

export type Station = {code: string; name: string; quality_status: string; retrospective: boolean;
  official_from: string; official_to: string; coverage_percent: number; latitude?: number; longitude?: number;
  altitude?: number; municipality?: string};
export type Catalog = {
  mode: string; version: string; reference: string; retrospective_period: string[];
  status?: string; catalog_sha256?: string; freeze_sha256?: string; dataset_sha256?: string;
  original_manifest_sha256?: string; source_run_id?: string; station_details?: Station[]; station_catalog_sha256?: string;
  formulas: Record<string, {label: string; inputs: string[]}>;
  methods: Record<string,string>; stations: Record<string,string>; calendars: string[];
  group_keys: Record<string,string[]>;
};
type Score = {method: string; n: number; mae: number | null; rmse: number | null; year?: string;
  bias: number | null; r2: number | null; coverage_calendar?: number | null;
  coefficient?: number; development_pairs?: number; retrospective_pairs?: number; cohort_id?: string;
  calendar_days?: number; pm_days?: number; predicted_days?: number; pm_pairs_own?: number; coverage_pm_own?: number | null};
type Scores = {rows: Score[]};
type Point = {date: string} & Record<string, string | number | null>;
const calendars: Record<string,string> = {annual:'Anual', seasonal:'Estacional', monthly:'Mensual'};
const groups: Record<string,string> = {global:'Global', station:'Estación', year:'Año',
  station_year:'Estación y año', month:'Mes climatológico', season:'Estación climática'};
const decimal = new Intl.NumberFormat('es-ES',{maximumFractionDigits:4});
const percent = new Intl.NumberFormat('es-ES',{style:'percent',maximumFractionDigits:1});
const fmt = (v?: number | null) => v == null ? '—' : decimal.format(v);

export function useRead<T>(path: string) {
  const [state, setState] = useState<{path: string; data?: T; error?: string}>();
  useEffect(() => {
    if (!path) return;
    const controller = new AbortController();
    fetch(path.startsWith('/') ? path : '/api/et0-v2/' + path,{signal:controller.signal})
      .then(async response => {
        if (response.status === 404) throw new Error('La vista previa no está habilitada.');
        if (!response.ok) throw new Error('No se pudo consultar esta selección.');
        return response.json() as Promise<T>;
      })
      .then(data => setState({path,data}))
      .catch(error => {
        if (!controller.signal.aborted) setState({path,error:error.message});
      });
    return () => controller.abort();
  }, [path]);
  return state?.path === path ? state : undefined;
}

export function Select({label, value, options, onChange}: {
  label: string; value: string; options: Record<string,string>; onChange: (value: string) => void;
}) {
  return <label>{label}<select value={value} onChange={e=>onChange(e.target.value)}>
    {Object.entries(options).map(([key,text])=><option value={key} key={key}>{text}</option>)}
  </select></label>;
}

function Results({rows, catalog, methods, inSample = false}: {
  rows: Score[]; catalog: Catalog; methods: string[]; inSample?: boolean;
}) {
  const [sort, setSort] = useState('rmse');
  if (!rows.length) return <p className="v2-empty">Sin resultados para esta selección.</p>;
  const eligible = rows.filter(row=>row.n > 0);
  const comparable = eligible.length === rows.length && eligible.length > 0 && eligible.every(row =>
    row.cohort_id && row.cohort_id === eligible[0].cohort_id && row.n === eligible[0].n &&
    (catalog.mode === 'synthetic' || (row.coverage_calendar != null &&
      row.coverage_calendar === eligible[0].coverage_calendar)));
  const metric = (row: Score) => {
    if (sort === 'bias') return row.bias == null ? Infinity : Math.abs(row.bias);
    if (sort === 'r2') return row.r2 == null ? Infinity : -row.r2;
    return (sort === 'mae' ? row.mae : row.rmse) ?? Infinity;
  };
  const ordered = comparable && !inSample ? [...rows].sort((a,b)=>metric(a)-metric(b)) : rows;
  const missing = inSample ? [] : methods.filter(method=>!rows.some(row=>row.method===method));
  return <>
    {!inSample && <div className="v2-table-context">
      <span>{comparable ? 'Ordenable · misma cohorte y cobertura' : 'Lectura paralela · consulte cohortes, cobertura y soporte'}</span>
      {comparable && <Select label="Ordenar por" value={sort} onChange={setSort}
        options={{rmse:'RMSE ↑',mae:'MAE ↑',bias:'Sesgo absoluto ↑',r2:'R² ↓'}}/>}
    </div>}
    <div className="v2-table" tabIndex={0} role="region" aria-label="Tabla de métricas">
      <table><caption>MAE, RMSE y sesgo en mm/día. {inSample ? 'Ajuste y evaluación sobre el mismo histórico.' : 'Cobertura propia sobre días de catálogo.'}</caption>
        <thead><tr><th scope="col">Método</th><th scope="col">n</th><th scope="col">MAE</th><th scope="col">RMSE</th>
          <th scope="col">Sesgo</th><th scope="col">R²</th>
          {inSample ? <><th scope="col">Coeficiente</th><th scope="col">Pares DEVELOPMENT</th><th scope="col">Pares 2020–2025</th></> :
            <th scope="col">Cobertura</th>}</tr></thead>
        <tbody>{ordered.map(row=><tr key={row.method}><th scope="row">{catalog.methods[row.method]}</th>
          <td>{row.n}</td><td>{fmt(row.mae)}</td><td>{fmt(row.rmse)}</td><td>{fmt(row.bias)}</td><td>{fmt(row.r2)}</td>
          {inSample ? <><td>{fmt(row.coefficient)}</td><td>{row.development_pairs}</td><td>{row.retrospective_pairs}</td></> :
            <td>{row.coverage_calendar == null ? '—' : percent.format(row.coverage_calendar)}</td>}</tr>)}</tbody>
      </table>
    </div>
    {missing.length > 0 && <p>Sin fila para {missing.map(method=>catalog.methods[method]).join(', ')}.</p>}
    {!inSample && <details><summary>Cohortes y denominadores</summary>
      <ul>{rows.map(row=><li key={row.method}>{catalog.methods[row.method]} · <code>{row.cohort_id}</code>
        {row.calendar_days != null && <> · {row.predicted_days}/{row.calendar_days} días de catálogo · {row.pm_pairs_own}/{row.pm_days} días con PM
          {row.coverage_pm_own != null && <> ({percent.format(row.coverage_pm_own)})</>}</>}</li>)}</ul>
    </details>}
  </>;
}

export function Workspace({catalog, initialStation, byFamily = false}: {catalog: Catalog; initialStation?: string; byFamily?: boolean}) {
  const real = catalog.mode !== 'synthetic';
  const [formula, setFormula] = useState('HG_RS_V1');
  const [inputMode, setInputMode] = useState('all');
  const [availableInputs, setAvailableInputs] = useState<string[]>([]);
  const inputNames = [...new Set(Object.values(catalog.formulas).flatMap(f => f.inputs))];
  const matchingFormulas = Object.entries(catalog.formulas).filter(([, f]) => inputMode === 'all' ||
    (f.inputs.every(input => availableInputs.includes(input)) &&
      (inputMode !== 'exact' || f.inputs.length === availableInputs.length)));
  const activeFormula = matchingFormulas.some(([id]) => id === formula) ? formula : matchingFormulas[0]?.[0];
  const [calendar, setCalendar] = useState('annual');
  const [view, setView] = useState('metrics');
  const [group, setGroup] = useState(initialStation ? 'station' : real ? 'global' : 'year');
  const [key, setKey] = useState(initialStation ?? (real ? 'all' : '2020'));
  const [compare, setCompare] = useState(real ? '' : '2021');
  const [station, setStation] = useState(initialStation ?? Object.keys(catalog.stations)[0]);
  const [start, setStart] = useState(real ? catalog.retrospective_period[0] : '2020-07-01');
  const [end, setEnd] = useState(real ? '2020-12-31' : '2020-07-08');
  const [methods, setMethods] = useState(byFamily ? ['AHC'] : Object.keys(catalog.methods).slice(0,5));
  const [cohort, setCohort] = useState('own');
  const [family, setFamily] = useState(byFamily ? 'AHC' : 'all');
  const base = new URLSearchParams({formula: activeFormula ?? formula,calendar});
  const metricPath = (selected: string) => {
    const query = new URLSearchParams(base);
    query.set('group',group);
    query.set('key',selected);
    methods.forEach(method=>query.append('method',method));
    return 'metrics?' + query;
  };
  const scores = useRead<Scores>(activeFormula && view==='metrics' ? metricPath(key) : '');
  const parallel = useRead<Scores>(view==='metrics' && compare && compare!==key ? metricPath(compare) : '');
  const canReadSeries = Boolean(start && end && start <= end);
  const seriesQuery = new URLSearchParams({formula: activeFormula ?? formula,calendar,station,start,end,cohort});
  methods.forEach(method=>seriesQuery.append('method',method));
  const series = useRead<{points: Point[]}>(['series','scatter'].includes(view) && canReadSeries ?
    'series?' + seriesQuery : '');
  const history = useRead<Scores>(view==='history' ? 'in-sample?' + new URLSearchParams({formula: activeFormula ?? formula,station}) : '');
  const trendQuery = new URLSearchParams(base);
  if (station) trendQuery.set('station',station);
  methods.forEach(method=>trendQuery.append('method',method));
  const evolution = useRead<Scores>(view==='evolution' ? 'evolution?' + trendQuery : '');
  const points = series?.data?.points ?? [];
  const labels = Object.fromEntries(catalog.group_keys[group].map(value=>[value,value==='all'?'Periodo completo':catalog.stations[value]??value]));
  const switchGroup = (value: string) => {
    setGroup(value);
    setKey(catalog.group_keys[value][0]);
    setCompare('');
  };
  return <>
    <section className="v2-panel" aria-label="Filtros de variables">
      <h3>Filtrar por variables disponibles</h3>
      <Select label="Criterio de variables" value={inputMode} onChange={setInputMode} options={{all:'Todas las fórmulas',exact:'Utiliza exactamente estas variables',subset:'Puede funcionar con estas variables'}}/>
      {inputMode !== 'all' && <fieldset><legend>Variables meteorológicas</legend>{inputNames.map(input => <label key={input}><input type="checkbox" checked={availableInputs.includes(input)} onChange={e=>setAvailableInputs(e.target.checked ? [...availableInputs,input] : availableInputs.filter(v=>v!==input))}/>{input}</label>)}</fieldset>}
      <p>El filtro selecciona fórmulas por sus entradas publicadas; no permite clasificar fórmulas distintas sin una cohorte común publicada.</p>
      {!activeFormula && <p role="status">Ninguna fórmula publicada coincide con las variables elegidas.</p>}
    </section>
    {activeFormula && <section className="v2-panel" aria-label="Selección">
      <div className="v2-filters">
        <Select label="Vista" value={view} onChange={next=>{setView(next);if(!station && next!=='evolution')setStation(Object.keys(catalog.stations)[0]);}} options={{metrics:'Comparar errores',series:'Series diarias',
          ...(real ? {scatter:'Estimación frente a PM',evolution:'Evolución anual del RMSE'} : {}),history:'AHC dentro de muestra'}}/>
        <Select label="Fórmula" value={activeFormula} onChange={setFormula}
          options={Object.fromEntries(matchingFormulas.map(([id,f])=>[id,f.label]))}/>
        {view!=='history' && <Select label="Calendario del modelo" value={calendar} onChange={setCalendar} options={calendars}/>}
        {view==='metrics' ? <>
          <Select label="Agrupar métricas por" value={group} onChange={switchGroup} options={groups}/>
          <Select label="Grupo principal" value={key} onChange={v=>{setKey(v); if(v===compare)setCompare('');}} options={labels}/>
          <Select label="Ver en paralelo" value={compare} onChange={setCompare} options={{'':'Sin segundo grupo',...Object.fromEntries(Object.entries(labels).filter(([id])=>id!==key))}}/>
        </> : <Select label="Estación" value={station} onChange={setStation}
          options={view==='evolution' ? {'':'Todas las estaciones',...catalog.stations} : catalog.stations}/>}
        {['series','scatter'].includes(view) && <><label>Desde<input type="date" value={start} onChange={e=>setStart(e.target.value)}/></label>
          <label>Hasta<input type="date" value={end} onChange={e=>setEnd(e.target.value)}/></label>
          {real && <Select label="Población de la serie" value={cohort} onChange={setCohort}
            options={{own:'Predicciones disponibles',...(methods.every(m=>!m.startsWith('ML_')) ? {common_core:'Fechas comunes a los cinco métodos principales'} : {})}}/>}</>}
      </div>
      <p>Entradas de {catalog.formulas[activeFormula].label}: {catalog.formulas[activeFormula].inputs.join(' · ')}.</p>
      {real && view!=='history' && <Select label="Familia de métodos" value={family}
        options={{all:'Todas',FORMULA:'Empírico',AHC:'AHC',ML:'ML clásico',ANN:'ANN'}}
        onChange={value=>{setFamily(value); setCohort('own'); setMethods(value==='all' ? Object.keys(catalog.methods).slice(0,5) :
          Object.keys(catalog.methods).filter(id=>id===value || id.startsWith(value+'_')));}}/>}
      {view!=='history' && <fieldset><legend>Métodos</legend>{Object.entries(catalog.methods)
        .filter(([id])=>family==='all' || id===family || id.startsWith(family+'_')).map(([id,label])=>
        <label key={id}><input type="checkbox" checked={methods.includes(id)} disabled={methods.length===1 && methods.includes(id)}
          onChange={e=>{if(id.startsWith('ML_'))setCohort('own'); setMethods(e.target.checked?[...methods,id]:methods.filter(m=>m!==id));}}/>{label}</label>)}</fieldset>}
    </section>}
    {activeFormula && view==='metrics' && <div className="v2-results">
      <section className="v2-panel"><h2>{groups[group]} · {labels[key]}</h2>
        {scores?.data ? <Results rows={scores.data.rows} methods={methods} catalog={catalog}/> :
          <p role="status">{scores?.error ?? 'Consultando métricas…'}</p>}</section>
      {compare && <section className="v2-panel"><h2>{groups[group]} · {labels[compare]}</h2>
        {parallel?.data ? <Results rows={parallel.data.rows} methods={methods} catalog={catalog}/> :
          <p role="status">{parallel?.error ?? 'Consultando métricas…'}</p>}</section>}
    </div>}
    {activeFormula && view==='series' && <section className="v2-panel"><h2>ET₀ diaria estimada y de referencia</h2>
      {points.length ? <>
        <Suspense fallback={<p role="status">Cargando gráfico…</p>}><Chart label="ET₀ diaria por método y referencia calculada" option={{
          legend:{type:'scroll',bottom:0}, tooltip:{trigger:'axis'}, grid:{left:55,right:20,bottom:65},
          xAxis:{type:'time'}, yAxis:{type:'value',name:'mm/día'},
          series:['reference',...methods].map((id,index)=>({name:catalog.methods[id]??'PM calculada',type:'line',
            symbol:['circle','rect','triangle','diamond','pin','arrow'][index%6],connectNulls:false,
            data:points.flatMap((p,i)=> i>0 && Date.parse(p.date)-Date.parse(points[i-1].date)>86400000 ?
              [[Date.parse(points[i-1].date)+86400000,null],[Date.parse(p.date),p[id]]] : [[Date.parse(p.date),p[id]]] )})),
        }}/></Suspense>
        <details><summary>Valores diarios en tabla</summary><div className="v2-table" tabIndex={0} role="region" aria-label="Serie diaria en mm/día">
          <table><thead><tr><th scope="col">Fecha</th><th scope="col">PM calculada</th>{methods.map(id=><th scope="col" key={id}>{catalog.methods[id]}</th>)}</tr></thead>
            <tbody>{points.map(p=><tr key={p.date}><th scope="row">{p.date}</th><td>{fmt(p.reference as number | null)}</td>
              {methods.map(id=><td key={id}>{fmt(p[id] as number | null)}</td>)}</tr>)}</tbody></table>
        </div></details>
      </> : <p role="status">{!canReadSeries ? 'Seleccione un intervalo válido.' : series?.error ??
        (series?.data ? 'Sin predicciones para esta selección.' : 'Consultando serie…')}</p>}
    </section>}
    {activeFormula && view==='scatter' && <section className="v2-panel"><h2>Estimación frente a PM calculada</h2>
      <p>Cada punto empareja la estimación con su referencia diaria. La diagonal indica igualdad.</p>
      {points.some(point=>point.reference != null && methods.some(method=>point[method]!=null)) ?
        <Suspense fallback={<p>Cargando gráfico…</p>}><Chart label="Dispersión de estimaciones frente a PM FAO56" option={scatterOptions(points, methods, catalog)}/></Suspense> :
        <p role="status">{!canReadSeries ? 'Seleccione un intervalo válido.' : series?.error ??
          (series?.data ? 'Sin pares estimación–PM para esta selección.' : 'Consultando valores diarios…')}</p>}
    </section>}
    {activeFormula && view==='evolution' && <section className="v2-panel"><h2>RMSE retrospectivo por año</h2>
      <p>Cada año representa su propia población evaluada. Las tablas de comparación muestran los denominadores de cada periodo.</p>
      {evolution?.data ? evolution.data.rows.some(row=>row.n > 0) ? <>
        <Suspense fallback={<p>Cargando gráfico…</p>}><Chart label="RMSE publicado por año y método" option={{
          legend:{type:'scroll',bottom:0}, tooltip:{trigger:'axis'}, grid:{left:55,right:20,bottom:65},
          xAxis:{type:'category',data:catalog.group_keys.year},yAxis:{type:'value',name:'mm/día'},
          series:methods.map(id=>({name:catalog.methods[id],type:'bar',data:catalog.group_keys.year.map(year=>
            evolution.data?.rows.find(row=>row.year===year && row.method===id)?.rmse ?? null)})),
        }}/></Suspense>
        <details><summary>Valores por año</summary>{catalog.group_keys.year.map(year=><section key={year}>
          <h3>{year}</h3><Results rows={evolution.data!.rows.filter(row=>row.year===year)} methods={methods} catalog={catalog}/>
        </section>)}</details>
      </> : <p className="v2-empty">Sin evaluación retrospectiva para esta selección.</p> :
        <p role="status">{evolution?.error ?? 'Consultando evolución…'}</p>}
    </section>}
    {activeFormula && view==='history' && <section className="v2-panel"><h2>AHC anual · histórico completo dentro de muestra</h2>
      <p>Los días evaluados también se usaron para ajustar el coeficiente.</p>
      {history?.data ? <Results rows={history.data.rows} methods={methods} catalog={catalog} inSample/> :
        <p role="status">{history?.error ?? 'Consultando histórico…'}</p>}
    </section>}
    {!real && <details className="v2-panel"><summary>Procedencia y versión</summary>
      <p>Versión de demostración <code>{catalog.version}</code>. Referencia prevista <code>{catalog.reference}</code>.
        Los valores de las tablas y los puntos de la serie son ejemplos separados.</p>
      <p>Los datos de estaciones del estudio proceden de SIAR. <a href="https://github.com/lependens/Capitulo-1">Repositorio del estudio</a></p>
    </details>}
  </>;
}

function scatterOptions(points: Point[], methods: string[], catalog: Catalog) {
  const plots = methods.map(id=>({name:catalog.methods[id],type:'scatter',symbolSize:7,
    data:points.flatMap(p=>typeof p.reference==='number' && typeof p[id]==='number' ? [[p.reference,p[id]]] : [])}));
  const values = plots.flatMap(plot=>plot.data.flat()) as number[];
  const min = Math.min(0,...values);
  const max = Math.max(1,...values);
  return {
    legend:{type:'scroll',bottom:0},tooltip:{trigger:'item'},grid:{left:63,right:25,bottom:85},
    xAxis:{type:'value',min,max,name:'PM calculada (mm/día)',nameLocation:'middle',nameGap:30},
    yAxis:{type:'value',min,max,name:'Estimación (mm/día)'},
    series:[...plots,{name:'Igualdad',type:'line',symbol:'none',data:[[min,min],[max,max]],lineStyle:{type:'dashed'}}],
  };
}

export default function Et0V2Preview() {
  const lookup = useRead<Catalog>('catalog');
  return <main className="v2-preview">
    <header className="v2-heading">
      <span>SIAR · Baleares</span><p className="v2-demo">Vista previa · Datos ficticios</p>
      <h1>Comparar modelos de ET₀ V2</h1>
      <p>Evaluación retrospectiva 2020–2025 · Referencia PM FAO56 calculada</p>
      <a href="/">Volver al explorador actual</a>
    </header>
    {lookup?.data?.mode === 'synthetic' ? <Workspace catalog={lookup.data}/> :
      <p role="status">{lookup?.error ?? (lookup?.data ? 'Esta instancia usa el catálogo real. Abra el explorador ET₀ V2.' : 'Cargando catálogo…')}</p>}
  </main>;
}
