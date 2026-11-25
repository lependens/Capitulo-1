import { lazy, Suspense, useState } from 'react';
import { Select, useRead, Workspace } from './Et0V2Preview';
import type { Catalog, Station } from './Et0V2Preview';

const Chart = lazy(() => import('./Chart'));
const sections = {
  home:'Inicio', stations:'Estaciones y datos', models:'Modelos y metodología',
  results:'Resultados por familia', compare:'Comparar modelos', sources:'Fuentes y auditoría',
};
type Section = keyof typeof sections;
const variables = {
  TempMedia:{label:'Temperatura media',unit:'°C'}, HumedadMedia:{label:'Humedad relativa media',unit:'%'},
  VelViento:{label:'Velocidad del viento',unit:'m/s'}, Radiacion:{label:'Radiación',unit:'MJ/m²/día'},
  ET0_PM_FAO56_V1:{label:'ET₀ PM FAO56 calculada',unit:'mm/día'},
};
type Variable = keyof typeof variables;

function StationMap({stations, current, select}: {stations: Station[]; current: string; select: (code:string)=>void}) {
  const located = stations.filter(s=>s.latitude != null && s.longitude != null);
  if (!located.length) return null;
  // A coordinate grid keeps the map usable on the LAN without external basemap requests.
  return <div className="v2-map" role="group" aria-label="Mapa de estaciones del catálogo SIAR">
    {located.map(station=><button key={station.code} className="v2-marker"
      style={{left:`${(station.longitude!-1)/3.45*100}%`,top:`${(40.22-station.latitude!)/1.74*100}%`}}
      aria-label={`${station.code} · ${station.name}`} aria-pressed={current===station.code}
      title={station.name} onClick={()=>select(station.code)}>{station.code.slice(2)}</button>)}
    <span className="v2-map-west">Ibiza y Formentera</span><span className="v2-map-center">Mallorca</span>
    <span className="v2-map-east">Menorca</span>
  </div>;
}

export function StationPage({catalog, current, select, compare}: {
  catalog: Catalog; current: string; select: (code:string)=>void; compare: ()=>void;
}) {
  const [variable, setVariable] = useState<Variable>('TempMedia');
  const [second, setSecond] = useState<Variable | ''>('');
  const [offset, setOffset] = useState(0);
  const station = catalog.station_details?.find(s=>s.code===current);
  const [start, setStart] = useState('');
  const [end, setEnd] = useState('');
  const query = new URLSearchParams({station:current, limit:'200', offset:String(offset)});
  if (start) query.set('from',start);
  if (end) query.set('to',end);
  const validRange = !start || !end || start <= end;
  const response = useRead<{rows: Record<string,string|null>[]}>(
    validRange ? '/api/observations?' + query : '');
  const rows = response?.data?.rows ?? [];
  const change = (code: string) => {setOffset(0); setStart(''); setEnd(''); select(code);};
  const metric = variables[variable];
  const yVariable = second ? variables[second] : null;
  const variableOptions = Object.fromEntries(Object.entries(variables).map(([id,v])=>[id,v.label]));
  return <>
    <section className="v2-panel"><h2>Estaciones y datos</h2>
      <StationMap stations={catalog.station_details??[]} current={current} select={change}/>
      {catalog.station_catalog_sha256 && <p className="v2-subtle">Posiciones del catálogo SIAR verificado. Mapa esquemático.</p>}
      <div className="v2-table" role="region" aria-label="Catálogo de estaciones" tabIndex={0}>
        <table><thead><tr><th>Estación</th><th>Periodo de catálogo</th><th>Quality V1</th><th>Retrospectiva V2</th></tr></thead>
          <tbody>{catalog.station_details?.map(s=><tr key={s.code}>
            <th scope="row"><button className="v2-link" onClick={()=>change(s.code)}>{s.code} · {s.name}</button></th>
            <td>{s.official_from} – {s.official_to}</td><td>{s.quality_status}</td>
            <td>{s.retrospective?'Disponible':'Serie histórica'}</td>
          </tr>)}</tbody></table>
      </div>
    </section>
    <section className="v2-panel"><h2>{current} · {catalog.stations[current]}</h2>
      <div className="v2-actions">
        <button onClick={compare}>Comparar modelos en esta estación</button>
        <button onClick={()=>window.print()}>Imprimir ficha</button>
      </div>
      {station && <p>{station.municipality} {station.altitude != null && <>· {station.altitude} m · </>}
        Cobertura de observaciones SIAR {station.coverage_percent}% · Quality V1 {station.quality_status}.</p>}
      {station && !station.retrospective && <p>Serie histórica sin observaciones para la evaluación V2 de 2020–2025.</p>}
      {current==='IB09' && <p>IB09 conserva Quality V1 FAIL. Su admisión excepcional se limita a DEVELOPMENT.</p>}
      <p>Disponibilidad variable × año, medias anuales y correlaciones: pendientes de tablas científicas publicadas con método, soporte y cobertura.</p>
      <div className="v2-filters">
        <Select label="Variable histórica" value={variable} options={variableOptions}
          onChange={id=>setVariable(id as Variable)}/>
        <Select label="Segunda variable" value={second} options={{'':'Serie temporal',...variableOptions}}
          onChange={id=>setSecond(id as Variable | '')}/>
        <label>Desde<input type="date" value={start} onChange={e=>{setOffset(0);setStart(e.target.value);}}/></label>
        <label>Hasta<input type="date" value={end} onChange={e=>{setOffset(0);setEnd(e.target.value);}}/></label>
      </div>
      <p>Observaciones del Dataset V1. Hasta 200 filas por página.</p>
      {rows.length > 0 ? <>
        <p>{rows[0].date} – {rows.at(-1)?.date}</p>
        <Suspense fallback={<p>Cargando gráfico…</p>}><Chart label={yVariable ?
          `${metric.label} frente a ${yVariable.label}` : `${metric.label} · ${metric.unit}`} option={{
          tooltip:{trigger:'item'}, grid:{left:57,right:20,bottom:50},
          xAxis:yVariable ? {type:'value',name:metric.unit,nameLocation:'middle',nameGap:28} : {type:'time'},
          yAxis:{type:'value',name:yVariable?.unit ?? metric.unit},
          series:[{type:'scatter',symbolSize:5,data:rows.flatMap(row=>row[variable] == null || row[variable]==='' ||
            (second && (row[second] == null || row[second]==='')) ? [] : second ?
            [[Number(row[variable]),Number(row[second])]] : [[Date.parse(row.date!), Number(row[variable])]])}],
        }}/></Suspense>
        <details><summary>Observaciones en tabla</summary><div className="v2-table" role="region" aria-label="Observaciones diarias" tabIndex={0}>
          <table><thead><tr><th>Fecha</th><th>{metric.label} ({metric.unit})</th>
            {yVariable && <th>{yVariable.label} ({yVariable.unit})</th>}</tr></thead>
            <tbody>{rows.map(row=><tr key={row.date}><th scope="row">{row.date}</th><td>{row[variable] || '—'}</td>
              {second && <td>{row[second] || '—'}</td>}</tr>)}</tbody></table>
        </div></details>
      </> : <p role="status">{!validRange ? 'Seleccione un intervalo válido.' :
        response?.error ?? (response?.data ? 'Sin observaciones para esta selección.' : 'Consultando observaciones…')}</p>}
      <div className="v2-actions"><button disabled={!offset} onClick={()=>setOffset(offset-200)}>Página anterior</button>
        <button disabled={rows.length<200} onClick={()=>setOffset(offset+200)}>Página siguiente</button></div>
    </section>
  </>;
}

function ModelPage({catalog}: {catalog: Catalog}) {
  const [formula,setFormula] = useState('HG_RS_V1');
  const [method,setMethod] = useState('AHC');
  const explanations: Record<string,string> = {
    FORMULA:'Cálculo empírico con las variables meteorológicas de la fórmula seleccionada.',
    AHC:'Ajuste por estación en DEVELOPMENT, mediante mínimos cuadrados por el origen. El coeficiente depende del calendario.',
    ML:'Modelo clásico seleccionado mediante validación por año omitido dentro de DEVELOPMENT.',
    ANN_A:'La validación interna usa el último 15 % cronológico del entrenamiento externo.',
    ANN_E:'La validación interna usa el 15 % de entrenamiento externo seleccionado de forma reproducible por hash.',
  };
  return <section className="v2-panel"><h2>Modelos y metodología</h2>
    <div className="v2-filters">
      <Select label="Fórmula e inputs" value={formula} onChange={setFormula}
        options={Object.fromEntries(Object.entries(catalog.formulas).map(([id,f])=>[id,f.label]))}/>
      <Select label="Método" value={method} onChange={setMethod} options={catalog.methods}/>
    </div>
    <h3>{catalog.methods[method]} · {catalog.formulas[formula].label}</h3>
    <p><code>{method}</code> · <code>{formula}</code></p>
    <dl><dt>Entradas meteorológicas</dt><dd>{catalog.formulas[formula].inputs.join(' · ')}</dd>
      <dt>Resoluciones</dt><dd>Anual, estacional y mensual. Estacional y mensual añaden indicadores de calendario.</dd>
      <dt>Procedimiento</dt><dd>{explanations[method] ?? 'Candidato ML clásico congelado por estación, fórmula y calendario.'}</dd>
      <dt>Evaluación</dt><dd>DEVELOPMENT 2004–2019. Retrospectiva 2020–2025 con configuraciones congeladas.</dd>
      <dt>Referencia</dt><dd>PM_FAO56_V1 calculada, no una medición instrumental independiente. Las métricas describen concordancia con esa referencia.</dd></dl>
    <p>El AHC de histórico completo se evalúa dentro de muestra. Algunos modelos no restringidos producen predicciones negativas, conservadas sin recorte. Las comparaciones requieren cohortes y coberturas compatibles. Las estaciones históricas tienen periodos más cortos; IB09 mantiene Quality V1 FAIL y su excepción se limita a DEVELOPMENT.</p>
    <button onClick={()=>window.print()}>Imprimir ficha</button>
  </section>;
}

function ExplorerV2({catalog}: {catalog: Catalog}) {
  const [section,setSection] = useState<Section>('home');
  const [station,setStation] = useState(Object.keys(catalog.stations)[0]);
  const [compareStation,setCompareStation] = useState<string>();
  const navigate = (next: Section) => {setCompareStation(undefined);setSection(next);};
  return <>
    <nav aria-label="Secciones del explorador" className="v2-nav">{Object.entries(sections).map(([id,label])=>
      <button key={id} aria-current={section===id?'page':undefined} onClick={()=>navigate(id as Section)}>{label}</button>)}</nav>
    {section==='home' && <section className="v2-panel"><h2>Explorar los resultados de Baleares</h2>
      <p>Consulta el histórico de estaciones SIAR, revisa las entradas de cada fórmula y compara estimaciones sobre la misma población.</p>
      <div className="v2-home-links">
        <button onClick={()=>navigate('stations')}>Explorar estaciones</button>
        <button onClick={()=>navigate('models')}>Ver metodología</button>
        <button onClick={()=>navigate('compare')}>Comparar estimaciones y errores</button>
      </div>
    </section>}
    {section==='stations' && <StationPage catalog={catalog} current={station} select={setStation}
      compare={()=>{setCompareStation(station);setSection('compare');}}/>}
    {section==='models' && <ModelPage catalog={catalog}/>}
    {(section==='results' || section==='compare') && <>
      <h2>{sections[section]}</h2>
      <Workspace key={section+compareStation} catalog={catalog} initialStation={compareStation}
        byFamily={section==='results'}/>
    </>}
    {section==='sources' && <section className="v2-panel"><h2>Fuentes y auditoría</h2>
      <p>Observaciones de estaciones procedentes de SIAR. La referencia diaria es PM FAO56 calculada.</p>
      <dl>{Object.entries({
        'Versión de resultados':catalog.version,'Integridad técnica':catalog.status,'Publicación científica':catalog.publication_status ?? 'no indicada','Ejecución':catalog.source_run_id,
        'Catálogo SHA‑256':catalog.catalog_sha256,'Manifiesto original SHA‑256':catalog.original_manifest_sha256,
        'Dataset SHA‑256':catalog.dataset_sha256,'Freeze SHA‑256':catalog.freeze_sha256,
        'Catálogo de estaciones SHA‑256':catalog.station_catalog_sha256,
      }).map(([key,value])=><div key={key}><dt>{key}</dt><dd><code>{value??'—'}</code></dd></div>)}</dl>
      <h3>Limitaciones científicas</h3><p>La referencia PM FAO56 es calculada y no constituye una medición instrumental independiente. El AHC de histórico completo es dentro de muestra. Algunas predicciones negativas de modelos no restringidos se conservan sin recorte. La comparación depende de cohortes y coberturas compatibles. IB07, IB09 e IB101 tienen periodos históricos limitados; IB09 conserva Quality V1 FAIL y su excepción se limita a DEVELOPMENT.</p><details><summary>Limitaciones publicadas en el catálogo</summary><ul>{catalog.limitations?.map((item)=><li key={item}>{item}</li>)}</ul></details><h3>Cómo leer las métricas</h3>
      <p>MAE indica el tamaño medio del error absoluto. RMSE da más peso a errores grandes. Un sesgo positivo indica sobreestimación.
        R² predictivo mide concordancia relativa y puede ser negativo. n cuenta pares evaluados. La cobertura indica disponibilidad respecto a días de catálogo.</p>
      <a href="https://github.com/lependens/Capitulo-1/pull/76">Documentación científica de la entrega</a>
    </section>}
  </>;
}

export default function Et0V2() {
  const lookup = useRead<Catalog>('catalog');
  return <main className="v2-preview">
    <header className="v2-heading"><span>SIAR · Baleares</span>
      {lookup?.data?.mode === 'review' && <p className="v2-demo">{lookup.data.status === 'VERIFIED' && lookup.data.publication_status === 'NOT_PUBLISHED' ? 'Resultados verificados · Pendientes de publicación científica' : `Entorno de revisión · ${lookup.data.status} · Publicación: ${lookup.data.publication_status ?? 'no indicada'}`}</p>}
      <h1>Explorador científico ET₀ V2</h1>
      <p>Retrospectiva 2020–2025 · Referencia PM FAO56 calculada</p>
    </header>
    {lookup?.data && lookup.data.mode !== 'synthetic' ? <ExplorerV2 catalog={lookup.data}/> :
      <p role="status">{lookup?.error ?? (lookup?.data ? 'Abra la vista previa sintética con ?preview=et0-v2.' : 'Consultando el catálogo…')}</p>}
  </main>;
}
