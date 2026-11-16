import { lazy, Suspense, useEffect, useState } from 'react';
import './et0-v2-preview.css';

const Chart = lazy(() => import('./Chart'));

type Catalog = {
  mode: string; version: string; reference: string; retrospective_period: string[];
  formulas: Record<string, {label: string; inputs: string[]}>;
  methods: Record<string,string>; stations: Record<string,string>; calendars: string[];
  group_keys: Record<string,string[]>;
};
type Score = {method: string; n: number; mae: number | null; rmse: number | null;
  bias: number | null; r2: number | null; coverage_calendar?: number | null;
  coefficient?: number; development_pairs?: number; retrospective_pairs?: number; cohort_id?: string};
type Scores = {rows: Score[]};
type Point = {date: string} & Record<string, string | number | null>;
const calendars: Record<string,string> = {annual:'Anual', seasonal:'Estacional', monthly:'Mensual'};
const groups: Record<string,string> = {global:'Global', station:'Estación', year:'Año',
  station_year:'Estación y año', month:'Mes climatológico', season:'Estación climática'};
const decimal = new Intl.NumberFormat('es-ES',{maximumFractionDigits:4});
const percent = new Intl.NumberFormat('es-ES',{style:'percent',maximumFractionDigits:1});
const fmt = (v?: number | null) => v == null ? '—' : decimal.format(v);

function useRead<T>(path: string) {
  const [state, setState] = useState<{path: string; data?: T; error?: string}>();
  useEffect(() => {
    if (!path) return;
    const controller = new AbortController();
    fetch('/api/et0-v2/' + path,{signal:controller.signal})
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

function Select({label, value, options, onChange}: {
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
  const comparable = eligible.length > 0 && eligible.every(row =>
    row.cohort_id && row.cohort_id === eligible[0].cohort_id && row.n === eligible[0].n);
  const metric = (row: Score) => {
    if (sort === 'bias') return row.bias == null ? Infinity : Math.abs(row.bias);
    if (sort === 'r2') return row.r2 == null ? Infinity : -row.r2;
    return (sort === 'mae' ? row.mae : row.rmse) ?? Infinity;
  };
  const ordered = comparable && !inSample ? [...rows].sort((a,b)=>metric(a)-metric(b)) : rows;
  const missing = inSample ? [] : methods.filter(method=>!rows.some(row=>row.method===method));
  return <>
    {!inSample && <div className="v2-table-context">
      <span>{comparable ? 'Métricas sobre las mismas fechas y estaciones' : 'Lectura paralela · cohortes distintas o sin soporte'}</span>
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
    {comparable && <details><summary>Cohorte de ejemplo</summary><code>{eligible[0].cohort_id}</code></details>}
  </>;
}

function Workspace({catalog}: {catalog: Catalog}) {
  const [formula, setFormula] = useState('HG_RS_V1');
  const [calendar, setCalendar] = useState('annual');
  const [view, setView] = useState('metrics');
  const [group, setGroup] = useState('year');
  const [key, setKey] = useState('2020');
  const [compare, setCompare] = useState('2021');
  const [station, setStation] = useState('DEMO-01');
  const [start, setStart] = useState('2020-07-01');
  const [end, setEnd] = useState('2020-07-08');
  const [methods, setMethods] = useState(Object.keys(catalog.methods));
  const base = new URLSearchParams({formula,calendar});
  const metricPath = (selected: string) => {
    const query = new URLSearchParams(base);
    query.set('group',group);
    query.set('key',selected);
    methods.forEach(method=>query.append('method',method));
    return 'metrics?' + query;
  };
  const scores = useRead<Scores>(view==='metrics' ? metricPath(key) : '');
  const parallel = useRead<Scores>(view==='metrics' && compare && compare!==key ? metricPath(compare) : '');
  const canReadSeries = Boolean(start && end && start <= end);
  const series = useRead<{points: Point[]}>(view==='series' && canReadSeries ?
    'series?' + new URLSearchParams({formula,calendar,station,start,end}) : '');
  const history = useRead<Scores>(view==='history' ? 'in-sample?' + new URLSearchParams({formula,station}) : '');
  const points = series?.data?.points ?? [];
  const labels = Object.fromEntries(catalog.group_keys[group].map(value=>[value,value==='all'?'Periodo completo':catalog.stations[value]??value]));
  const switchGroup = (value: string) => {
    setGroup(value);
    setKey(catalog.group_keys[value][0]);
    setCompare('');
  };
  return <>
    <section className="v2-panel" aria-label="Selección">
      <div className="v2-filters">
        <Select label="Vista" value={view} onChange={setView} options={{metrics:'Comparar errores',series:'Series diarias',history:'AHC dentro de muestra'}}/>
        <Select label="Fórmula" value={formula} onChange={setFormula}
          options={Object.fromEntries(Object.entries(catalog.formulas).map(([id,f])=>[id,f.label]))}/>
        {view!=='history' && <Select label="Calendario del modelo" value={calendar} onChange={setCalendar} options={calendars}/>}
        {view==='metrics' ? <>
          <Select label="Agrupar métricas por" value={group} onChange={switchGroup} options={groups}/>
          <Select label="Grupo principal" value={key} onChange={v=>{setKey(v); if(v===compare)setCompare('');}} options={labels}/>
          <Select label="Ver en paralelo" value={compare} onChange={setCompare} options={{'':'Sin segundo grupo',...Object.fromEntries(Object.entries(labels).filter(([id])=>id!==key))}}/>
        </> : <Select label="Estación" value={station} onChange={setStation} options={catalog.stations}/>}
        {view==='series' && <><label>Desde<input type="date" value={start} onChange={e=>setStart(e.target.value)}/></label>
          <label>Hasta<input type="date" value={end} onChange={e=>setEnd(e.target.value)}/></label></>}
      </div>
      <p>Entradas de {catalog.formulas[formula].label}: {catalog.formulas[formula].inputs.join(' · ')}.</p>
      {view!=='history' && <fieldset><legend>Métodos</legend>{Object.entries(catalog.methods).map(([id,label])=>
        <label key={id}><input type="checkbox" checked={methods.includes(id)} disabled={methods.length===1 && methods.includes(id)}
          onChange={e=>setMethods(e.target.checked?[...methods,id]:methods.filter(m=>m!==id))}/>{label}</label>)}</fieldset>}
    </section>
    {view==='metrics' && <div className="v2-results">
      <section className="v2-panel"><h2>{groups[group]} · {labels[key]}</h2>
        {scores?.data ? <Results rows={scores.data.rows} methods={methods} catalog={catalog}/> :
          <p role="status">{scores?.error ?? 'Consultando métricas…'}</p>}</section>
      {compare && <section className="v2-panel"><h2>{groups[group]} · {labels[compare]}</h2>
        {parallel?.data ? <Results rows={parallel.data.rows} methods={methods} catalog={catalog}/> :
          <p role="status">{parallel?.error ?? 'Consultando métricas…'}</p>}</section>}
    </div>}
    {view==='series' && <section className="v2-panel"><h2>ET₀ diaria estimada y de referencia</h2>
      {points.length ? <>
        <Suspense fallback={<p role="status">Cargando gráfico…</p>}><Chart label="ET₀ diaria por método y referencia calculada" option={{
          legend:{type:'scroll',bottom:0}, tooltip:{trigger:'axis'}, grid:{left:55,right:20,bottom:65},
          xAxis:{type:'category',data:points.map(p=>p.date)}, yAxis:{type:'value',name:'mm/día'},
          series:['reference',...methods].map((id,index)=>({name:catalog.methods[id]??'PM calculada',type:'line',
            symbol:['circle','rect','triangle','diamond','pin','arrow'][index],connectNulls:false,
            data:points.map(p=>p[id])})),
        }}/></Suspense>
        <details><summary>Valores diarios en tabla</summary><div className="v2-table" tabIndex={0} role="region" aria-label="Serie diaria en mm/día">
          <table><thead><tr><th scope="col">Fecha</th><th scope="col">PM calculada</th>{methods.map(id=><th scope="col" key={id}>{catalog.methods[id]}</th>)}</tr></thead>
            <tbody>{points.map(p=><tr key={p.date}><th scope="row">{p.date}</th><td>{fmt(p.reference as number | null)}</td>
              {methods.map(id=><td key={id}>{fmt(p[id] as number | null)}</td>)}</tr>)}</tbody></table>
        </div></details>
      </> : <p role="status">{!canReadSeries ? 'Seleccione un intervalo válido.' : series?.error ??
        (series?.data ? 'Sin puntos de ejemplo. Pruebe Hargreaves Rs, anual, DEMO-01 y julio de 2020.' : 'Consultando serie…')}</p>}
    </section>}
    {view==='history' && <section className="v2-panel"><h2>AHC anual · histórico completo dentro de muestra</h2>
      <p>Los días evaluados también se usaron para ajustar el coeficiente.</p>
      {history?.data ? <Results rows={history.data.rows} methods={methods} catalog={catalog} inSample/> :
        <p role="status">{history?.error ?? 'Consultando histórico…'}</p>}
    </section>}
    <details className="v2-panel"><summary>Procedencia y versión</summary>
      <p>Versión de demostración <code>{catalog.version}</code>. Referencia prevista <code>{catalog.reference}</code>.
        Los valores de las tablas y los puntos de la serie son ejemplos separados.</p>
      <p>Los datos de estaciones del estudio proceden de SIAR. <a href="https://github.com/lependens/Capitulo-1">Repositorio del estudio</a></p>
    </details>
  </>;
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
    {lookup?.data ? <Workspace catalog={lookup.data}/> : <p role="status">{lookup?.error ?? 'Cargando catálogo…'}</p>}
  </main>;
}
