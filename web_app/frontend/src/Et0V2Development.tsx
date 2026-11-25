import {lazy, Suspense, useState} from 'react';
import type {ReactNode} from 'react';
import {Select, useRead} from './Et0V2Preview';
import type {Catalog} from './Et0V2Preview';
import {fmt, methodColor, methodName} from './Et0V2Compare';
import {calendarNames} from './et0Compare';

const Chart=lazy(()=>import('./Chart'));
type Page='ahc'|'ml'|'ann';
type Row=Record<string,string|number|null>;
type Configuration={algorithm?:string;parameters?:Record<string,unknown>;architecture?:number[];parameter_count?:number};
type Candidate={candidate_id:string;configuration:Configuration;inputs:string[];fold_count:number;n:number;mean_annual_rmse:number;
  sd_annual_rmse:number;pooled_rmse:number;is_principal:boolean;selected_by_algorithm:boolean;warnings?:unknown};
type Final={candidate_id:string;model_label:string;model_sha256:string;frozen_sha256:string;selected_configuration:Configuration;
  is_principal:boolean;seeds?:number[];epochs?:Record<string,number>};
type Cell={station_code:string;family:string;calendar:string;ahc_available:boolean;ml_available:boolean;ann_a_available:boolean;ann_e_available:boolean;
  eligible_paired_days:number;effective_first_date:string|null;effective_last_date:string|null;reason:string|null};
type Frequency={group:string;choice:string;count:number};
type Rows<T>={rows:T[];total:number;table:string};
type Source={catalog_sha256:string;dataset_sha256:string;freeze_sha256:string;integrity:string;publication_status:string;
  tables:Record<string,{rows:number;bytes:number;sha256:string}>};
export type DevelopmentCatalog={source:Source;cells:Cell[]};
export const variants=['LINEAR','RIDGE','ELASTIC_NET','TREE','RANDOM_FOREST','EXTRA_TREES','HIST_GRADIENT_BOOSTING'];
const details=(value:unknown)=>value==null?'—':JSON.stringify(value);
const numeric=(row:Row,key:string)=>typeof row[key]==='number'?fmt(row[key] as number):'—';
const displayGroup=(group:string)=>({annual:'Año completo',DJF:'Invierno · DJF',MAM:'Primavera · MAM',JJA:'Verano · JJA',SON:'Otoño · SON'}[group]??group);
const names=(catalog:Catalog)=>Object.fromEntries(Object.entries(catalog.stations).map(([code,name])=>[code,code+' · '+name]));
export function Table({columns,rows}:{columns:string[];rows:ReactNode[][]}) {
  return <div className="v2-table" role="region" aria-label={columns.join(', ')} tabIndex={0}><table>
    <thead><tr>{columns.map(c=><th key={c} scope="col">{c}</th>)}</tr></thead>
    <tbody>{rows.map((row,i)=><tr key={i}>{row.map((value,j)=>j===0?<th key={j} scope="row">{value}</th>:<td key={j}>{value??'—'}</td>)}</tr>)}</tbody>
  </table></div>;
}
function DataState({data,error,empty='Sin filas para esta selección.'}:{data:unknown;error?:string;empty?:string}) {
  return <p role="status">{error??(data?empty:'Consultando datos verificados…')}</p>;
}
function useTable<T>(name:string,filters:Record<string,string>,enabled=true) {
  return useRead<Rows<T>>(enabled?'development/table/'+name+'?'+new URLSearchParams({...filters,limit:'150'}):'');
}
function Bars({title,labels,series,vertical=true}:{title:string;labels:string[];vertical?:boolean;
  series:{name:string;data:(number|null)[];color?:string}[]}) {
  const category={type:'category',data:labels};
  const measure={type:'value',name:vertical?title:''};
  return <Suspense fallback={<p>Cargando gráfico…</p>}><Chart label={title} option={{
    tooltip:{trigger:'axis'},legend:series.length>1?{type:'scroll',bottom:0}:undefined,
    grid:{left:vertical?65:140,right:26,bottom:series.length>1?62:44,top:36},
    xAxis:vertical?category:measure,yAxis:vertical?measure:category,
    dataZoom:labels.length>14?[{type:'inside',orient:vertical?'horizontal':'vertical'}]:undefined,
    animation:false,series:series.map(s=>({name:s.name,data:s.data,type:'bar',itemStyle:s.color?{color:s.color}:undefined})),
  }}/></Suspense>;
}
function Frequencies({catalog,stage,station,family,calendar}:{catalog:Catalog;stage:string;station:string;family:string;calendar:string}) {
  const [dimension,setDimension]=useState('global');
  const [limited,setLimited]=useState(false);
  const [open,setOpen]=useState(false);
  const params=new URLSearchParams({stage,dimension});
  if(limited) {
    if(dimension!=='station_code')params.set('station',station);
    if(dimension!=='family')params.set('family',family);
    if(dimension!=='calendar')params.set('calendar',calendar);
  }
  const request=useRead<{rows:Frequency[];total:number}>(open?'development/frequency?'+params:'');
  const rows=request?.data?.rows??[];
  const label=(key:string)=>catalog.formulas[key]?.label??calendarNames[key]??(catalog.stations[key]?key+' · '+catalog.stations[key]:key);
  const choice=(key:string)=>stage==='ML'?methodName('ML_'+key):key;
  const groups=[...new Set(rows.map(row=>row.group))];
  const choices=stage==='ML'?variants:['16','32','64','64–32','64–32–16'];
  const multiple=dimension!=='global';
  const count=(group:string,key:string)=>rows.find(r=>r.group===group&&r.choice===key)?.count??0;
  const selection=multiple?groups.flatMap(group=>choices.map(key=>[label(group),choice(key),count(group,key)])):
    choices.map(key=>[choice(key),count(groups[0],key)]);
  return <details onToggle={e=>setOpen(e.currentTarget.open)}><summary>Frecuencia de selección en DEVELOPMENT</summary>
    <div className="p2-filters"><Select label="Desglosar frecuencia por" value={dimension} onChange={setDimension}
      options={{global:'Total',station_code:'Estación',family:'Familia',calendar:'Configuración'}}/>
      <label className="p2-inline"><input type="checkbox" checked={limited} onChange={e=>setLimited(e.target.checked)}/> Aplicar otros filtros de la ficha</label>
    </div>
    <p>Se cuenta una selección {stage==='ML'?'del ML principal':'de la arquitectura del ensemble'} por celda compatible. {request?.data&&request.data.total+' celdas en esta consulta.'}</p>
    {rows.length?<><Bars title="Celdas seleccionadas" vertical={multiple}
      labels={multiple?groups.map(label):choices.map(choice)}
      series={multiple?choices.map(key=>({name:choice(key),data:groups.map(group=>count(group,key))})):
        [{name:'Celdas seleccionadas',data:choices.map(key=>count(groups[0],key)),color:methodColor(stage)}]}/>
      <Table columns={multiple?['Grupo','Selección','Celdas']:['Selección','Celdas']} rows={selection}/></>:
      <DataState data={request?.data} error={request?.error}/>}
  </details>;
}
function Calibration({station,family,calendar}:{station:string;family:string;calendar:string}) {
  const [showFolds,setShowFolds]=useState(false);
  const [foldYear,setFoldYear]=useState('all');
  const [showHistory,setShowHistory]=useState(false);
  const base={station,family};
  const response=useTable<Row>('ahc-frozen-coefficients',base);
  const folds=useTable<Row>('ahc-fold-coefficients',{...base,calendar,...(foldYear==='all'?{}:{year:foldYear})},showFolds);
  const history=useTable<Row>('ahc-history-in-sample',base,showHistory);
  const rows=response?.data?.rows??[];
  const selected=rows.filter(r=>r.calendar===calendar);
  const months=Array.from({length:12},(_,i)=>String(i+1).padStart(2,'0'));
  const season=['DJF','DJF','MAM','MAM','MAM','JJA','JJA','JJA','SON','SON','SON','DJF'];
  const bands=Object.keys(calendarNames).map(k=>({
    name:calendarNames[k],data:months.map((m,i)=>{
      const group=k==='annual'?'annual':k==='seasonal'?season[i]:m;
      return (rows.find(r=>r.calendar===k&&r.group===group)?.coefficient as number|null)??null;
    }),
  }));
  return <>
    <section className="p2-panel"><h2>DEVELOPMENT · coeficientes congelados</h2>
      <p className="p2-equation">ET₀ ajustada = c × ET₀ de la fórmula</p>
      <p>El coeficiente c minimiza el error cuadrático respecto a PM sobre los pares válidos. El gráfico repite cada coeficiente anual o estacional en sus meses de aplicación.</p>
      {rows.length?<><Bars title="Coeficiente AHC por mes de aplicación" labels={months} series={bands}/>
        <Table columns={['Configuración','Grupo','Pares','Σx²','Coeficiente','Ventana de ajuste','Soporte']} rows={rows.map(r=>[
          calendarNames[String(r.calendar)],displayGroup(String(r.group)),r.n,numeric(r,'sum_x2'),numeric(r,'coefficient'),
          r.fit_start+' a '+r.fit_end,r.support==='IDENTIFIABLE'?'Identificable':'Grupo sin coeficiente',
        ])}/><p>La ventana contractual 2004–2019 puede incluir días sin pares. La tabla muestra n por grupo.</p></>:
        <DataState data={response?.data} error={response?.error}/>}
      <details onToggle={e=>setShowFolds(e.currentTarget.open)}><summary>Coeficientes por año excluido · {calendarNames[calendar]}</summary>
        <Select label="Año excluido" value={foldYear} onChange={setFoldYear}
          options={{all:'Todos',...Object.fromEntries(Array.from({length:16},(_,i)=>[2004+i,String(2004+i)]))}}/>
        {folds?.data?.rows.length?<><Table columns={['Año excluido','Grupo','Pares','Σx²','Coeficiente']} rows={folds.data.rows.map(r=>[
          r.held_out_year,displayGroup(String(r.group)),r.n,numeric(r,'sum_x2'),numeric(r,'coefficient'),
        ])}/>{folds.data.total>folds.data.rows.length&&<p>Mostrando {folds.data.rows.length} de {folds.data.total}. Selecciona un año para ver sus grupos completos.</p>}</>:
          <DataState data={folds?.data} error={folds?.error}/>}
      </details>
      <p className="v2-subtle">Fuentes: ahc-frozen-coefficients.csv y ahc-fold-coefficients.csv. {selected.length} grupos congelados para {calendarNames[calendar].toLowerCase()}.</p>
    </section>
    <details className="p2-panel" onToggle={e=>setShowHistory(e.currentTarget.open)}><summary>Histórico completo · dentro de muestra</summary>
      <p>Esta vista usa un ajuste anual que incluye los días evaluados en la misma serie.</p>
      {history?.data?.rows.length?<Table columns={['Periodo','Coeficiente','Pares','Pares DEVELOPMENT','Pares retrospectivos','RMSE','MAE','Sesgo','R²']}
        rows={history.data.rows.map(r=>[r.years,numeric(r,'coefficient'),r.n,r.development_pairs,r.retrospective_pairs,numeric(r,'rmse'),numeric(r,'mae'),numeric(r,'bias'),numeric(r,'r2')])}/>:
        <DataState data={history?.data} error={history?.error}/>}
      <p className="v2-subtle">Fuente: ahc-history-in-sample.csv.</p>
    </details>
  </>;
}
function Architecture({candidate}:{candidate:Candidate}) {
  const layers=[candidate.inputs.length,...(candidate.configuration.architecture??[]),1];
  const centers=(nodes:number)=>Array.from({length:Math.min(5,nodes)},(_,i)=>95+(i-(Math.min(5,nodes)-1)/2)*24);
  const x=(i:number)=>65+i*530/(layers.length-1);
  return <div className="p2-network"><svg viewBox="0 0 660 190" role="img"
    aria-label={'Capas densas '+layers.join(' → ')+'; las capas grandes están representadas mediante nodos muestrales'}>
    {layers.slice(1).map((nodes,i)=>centers(nodes).flatMap((y,j)=>centers(layers[i]).map((start,k)=>
      <line key={i+':'+j+':'+k} x1={x(i)} y1={start} x2={x(i+1)} y2={y} stroke="#a2bbb8" strokeWidth="1"/>)))}
    {layers.map((nodes,i)=><g key={i}>{centers(nodes).map(y=><circle key={y} cx={x(i)} cy={y} r="7" fill="#7151b1"/>)}
      <text textAnchor="middle" x={x(i)} y="24" fill="currentColor" fontSize="15">{nodes} {i===0?'entradas':i===layers.length-1?'salida':'ReLU'}</text>
    </g>)}
  </svg><p>{candidate.configuration.parameter_count} parámetros por red. Entradas: {candidate.inputs.join(' · ')}.</p></div>;
}
function FoldTable({kind,filters,candidate}:{kind:'ml'|'ann';filters:Record<string,string>;candidate:string}) {
  const [open,setOpen]=useState(false);
  const [seedMode,setSeedMode]=useState(false);
  const response=useTable<Row>(kind==='ml'?'ml-fold-metrics':'ann-fold-seed-metrics',{...filters,candidate},open&&!!candidate);
  const rows=response?.data?.rows??[];
  const population=kind==='ann'&&!seedMode?[...new Map(rows.map(row=>[row.year,row])).values()]:rows;
  const prefix=kind==='ann'?(seedMode?'seed_':'ensemble_'):'';
  return <details onToggle={e=>setOpen(e.currentTarget.open)}><summary>Métricas por fold del candidato</summary>
    {kind==='ann'&&<><label className="p2-inline"><input type="checkbox" checked={seedMode} onChange={e=>setSeedMode(e.target.checked)}/> Ver métricas individuales por semilla</label>
      <p>La vista de ensemble muestra un resultado por año excluido. La vista de semillas muestra cada red y sus épocas.</p></>}
    {population.length?<Table columns={['Año excluido',...(kind==='ann'&&seedMode?['Semilla','Mejor época','Épocas ejecutadas']:[]),'Pares','RMSE','MAE','Sesgo','R²']}
      rows={population.map(r=>[r.year,...(kind==='ann'&&seedMode?[r.seed_seed,r.seed_best_epoch,r.seed_epochs_run]:[]),
        r[kind==='ann'&&seedMode?'seed_n':'n'],...['rmse','mae','bias','r2'].map(m=>numeric(r,prefix+m))])}/>:
        <DataState data={response?.data} error={response?.error}/>}
    <p className="v2-subtle">Fuente: {kind==='ml'?'ml-fold-metrics.csv':'ann-fold-seed-metrics.csv'}.</p>
  </details>;
}
function Candidates({kind,filters,supported}:{kind:'ml'|'ann';filters:Record<string,string>;supported:boolean}) {
  const [algorithm,setAlgorithm]=useState('all');
  const [candidateId,setCandidateId]=useState('');
  const [comparison,setComparison]=useState('mean_annual_rmse');
  const candidates=useTable<Candidate>(kind+'-development-candidates',filters,supported);
  const finals=useTable<Final>(kind==='ml'?'ml-final-models':'ann-selected-architectures',filters,supported);
  const all=candidates?.data?.rows??[];
  const frozen=finals?.data?.rows??[];
  const filtered=all.filter(c=>algorithm==='all'||kind==='ann'||c.configuration.algorithm===algorithm);
  const item=filtered.find(row=>row.candidate_id===candidateId)??filtered.find(row=>row.is_principal)??filtered.find(row=>row.selected_by_algorithm)??filtered[0];
  const algorithmName=(row:Candidate)=>kind==='ann'?row.configuration.architecture?.join('–')??'—':methodName('ML_'+row.configuration.algorithm);
  const candidateLabel=(row:Candidate)=>algorithmName(row)+(kind==='ml'?' · '+details(row.configuration.parameters):'');
  const principal=frozen.find(row=>row.is_principal);
  if(!supported)return <p role="status">Esta celda carece de suficientes folds externos con soporte para {kind==='ml'?'ML':'ANN'}. El catálogo conserva el ajuste AHC donde sus grupos son identificables.</p>;
  if(!all.length)return <DataState data={candidates?.data} error={candidates?.error}/>;
  return <>
    <p>{kind==='ml'?'Los 37 candidatos cubren siete algoritmos. Cada algoritmo aporta su configuración congelada y uno se designa ML principal.':'Cinco arquitecturas por variante. ANN A valida con el último 15 % cronológico y ANN E con un 15 % reproducible por hash.'}</p>
    {principal&&<p><strong>{kind==='ml'?'ML principal':'Arquitectura seleccionada'}</strong> · {kind==='ml'?methodName('ML_'+principal.selected_configuration.algorithm):principal.selected_configuration.architecture?.join('–')}</p>}
    <div className="p2-filters">
      {kind==='ml'&&<Select label="Algoritmo de candidatos" value={algorithm} onChange={setAlgorithm}
        options={{all:'Los siete algoritmos',...Object.fromEntries(variants.map(v=>[v,methodName('ML_'+v)]))}}/>}
      <Select label={kind==='ml'?'Candidato para inspeccionar':'Arquitectura para inspeccionar'} value={item?.candidate_id??''} onChange={setCandidateId}
        options={Object.fromEntries(filtered.map(row=>[row.candidate_id,candidateLabel(row)]))}/>
      <Select label="Métrica DEVELOPMENT" value={comparison} onChange={setComparison}
        options={{mean_annual_rmse:'Media de RMSE por fold',pooled_rmse:'RMSE agrupado'}}/>
    </div>
    {item&&kind==='ann'&&<Architecture candidate={item}/>}
    <Bars title="RMSE DEVELOPMENT · mm/día" labels={filtered.map((row,i)=>kind==='ml'?'C'+(i+1):algorithmName(row))}
      series={[{name:'RMSE',color:methodColor(filters.stage),data:filtered.map(r=>comparison==='pooled_rmse'?r.pooled_rmse:r.mean_annual_rmse)}]}/>
    <Table columns={['Candidato','Configuración','Folds','Pares','Media RMSE','DE RMSE','RMSE agrupado','Selección']} rows={filtered.map((r,i)=>[
      kind==='ml'?'C'+(i+1)+' · '+algorithmName(r):algorithmName(r),kind==='ml'?details(r.configuration.parameters):r.configuration.parameter_count,
      r.fold_count,r.n,fmt(r.mean_annual_rmse),fmt(r.sd_annual_rmse),fmt(r.pooled_rmse),
      r.is_principal?'Principal':r.selected_by_algorithm?'Congelado por algoritmo':'Candidato',
    ])}/>
    {item&&<><p>Ficha del candidato: <code className="p2-hash">{item.candidate_id}</code>. Entradas: {item.inputs.join(' · ')}.</p>
      {kind==='ml'&&item.warnings!=null&&<details><summary>Avisos conservados del candidato</summary><p>Los avisos son evidencia del proceso de ajuste. Su recuento no mide exactamente los entrenamientos que dejaron de converger.</p><p>{details(item.warnings)}</p></details>}
      <FoldTable kind={kind} filters={filters} candidate={item.candidate_id}/></>}
    <details><summary>{kind==='ml'?'Estimadores finales y procedencia':'Ensemble congelado y épocas finales'}</summary>
      <Table columns={['Modelo final','Candidato','Configuración','Épocas por semilla','Hash del modelo']} rows={frozen.map(r=>[
        r.model_label+(r.is_principal?' · principal':''),r.candidate_id,details(r.selected_configuration),
        r.seeds?.map(seed=>seed+': '+(r.epochs?.[String(seed)]??'—')).join(' · ')??'—',
        <abbr title={r.model_sha256}>{r.model_sha256.slice(0,12)}</abbr>,
      ])}/>
      <p className="v2-subtle">Fuentes: {kind==='ml'?'ml-development-candidates.csv y ml-final-models.csv':'ann-development-candidates.csv y ann-selected-architectures.csv'}.</p>
    </details>
  </>;
}
export function Development({catalog,page,initialStation}:{catalog:Catalog;page:Page;initialStation:string}) {
  const [station,setStation]=useState(initialStation);
  const [family,setFamily]=useState('HG_RS_V1');
  const [calendar,setCalendar]=useState('annual');
  const [variant,setVariant]=useState<'ANN_A'|'ANN_E'>('ANN_A');
  const metadata=useRead<DevelopmentCatalog>('development/catalog');
  const stage=page==='ahc'?'AHC':page==='ml'?'ML':variant;
  const supportKey=stage.toLowerCase()+'_available' as 'ahc_available'|'ml_available'|'ann_a_available'|'ann_e_available';
  const cells=metadata?.data?.cells.filter(row=>row.station_code===station&&row.family===family)??[];
  const cell=cells.find(row=>row.calendar===calendar);
  const filters={station,family,calendar,stage};
  const testKey=catalog.group_keys.station?.includes(station)?station:null;
  const testQuery=new URLSearchParams({formula:family,group:'station',key:testKey??''});
  const testCalendars=page==='ahc'?Object.keys(calendarNames):[calendar];
  const methods=page==='ahc'?['AHC','FORMULA']:page==='ml'?['ML',...variants.map(v=>'ML_'+v)]:['ANN_A','ANN_E'];
  testCalendars.forEach(k=>testQuery.append('calendar',k));
  methods.forEach(k=>testQuery.append('method',k));
  const test=useRead<Rows<Row>>(testKey?'comparison?'+testQuery:'');
  return <>
    <section className="p2-panel"><div className="p2-filters">
      <Select label="Estación" options={names(catalog)} value={station} onChange={setStation}/>
      <Select label="Familia de entradas" options={Object.fromEntries(Object.entries(catalog.formulas).map(([id,f])=>[id,f.label]))} value={family} onChange={setFamily}/>
      <label>Configuración del modelo<select value={calendar} onChange={e=>setCalendar(e.target.value)}>
        {Object.entries(calendarNames).map(([key,title])=>{
          const available=cells.find(row=>row.calendar===key)?.[supportKey];
          return <option key={key} value={key} disabled={available===false}>{title}{available===false?' · sin soporte':''}</option>;
        })}
      </select></label>
      {page==='ann'&&<Select label="Variante ANN" options={{ANN_A:'ANN A · validación cronológica',ANN_E:'ANN E · validación por hash'}} value={variant} onChange={v=>setVariant(v as typeof variant)}/>}
    </div>
    <p>Entradas: {catalog.formulas[family]?.inputs.join(' · ')}.</p>
    {cell&&<p>DEVELOPMENT efectivo: {cell.effective_first_date??'—'} a {cell.effective_last_date??'—'} · {cell.eligible_paired_days} días elegibles.</p>}
    </section>
    {page==='ahc'?<Calibration station={station} family={family} calendar={calendar}/>:
      <section className="p2-panel"><h2>DEVELOPMENT · candidatos y selección congelada</h2>
        {cell?<Candidates key={[station,family,calendar,stage].join('/')} kind={page} filters={filters} supported={cell[supportKey]}/>:
          <DataState data={metadata?.data} error={metadata?.error?'No se pudo leer el catálogo complementario DEVELOPMENT.':undefined}/>}
        <Frequencies catalog={catalog} stage={stage} station={station} family={family} calendar={calendar}/>
      </section>}
    <section className="p2-panel"><h2>TEST retrospectivo · 2020–2025</h2>
      <p>{page==='ahc'?'La evaluación muestra la fórmula original y el AHC congelado para cada calendario.':
        'Las filas corresponden exclusivamente a los modelos congelados tras DEVELOPMENT. La selección de un candidato arriba abre sus validaciones internas.'}</p>
      {test?.data?.rows.some(r=>Number(r.n)>0)?<Table columns={['Método','Calendario','Pares','RMSE','MAE','Sesgo','R²','Cobertura']}
        rows={test.data.rows.map(r=>[methodName(String(r.method)),calendarNames[String(r.calendar)],r.n,
          ...['rmse','mae','bias','r2'].map(m=>numeric(r,m)),typeof r.coverage_calendar==='number'?fmt(100*r.coverage_calendar)+' %':'—'])}/>:
        <p role="status">{test?.error??(!testKey||test?.data?'Esta estación carece de TEST para esta selección.':'Consultando TEST…')}</p>}
      <p className="v2-subtle">Fuente: retrospective-metrics.csv · catálogo retrospectivo · referencia PM_FAO56_V1 calculada.</p>
    </section>
  </>;
}
