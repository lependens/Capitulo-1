import {lazy, Suspense, useEffect, useState} from 'react';
import {StationPage} from './Et0V2';
import {Select, useRead, Workspace} from './Et0V2Preview';
import type {Catalog} from './Et0V2Preview';
import {Compare, fmt, methodName, methodColor} from './Et0V2Compare';
import type {Preset} from './Et0V2Compare';
import {calendarNames} from './et0Compare';
import memo from './et0-v2-memo.json';
import './et0-phase2.css';

const Chart = lazy(()=>import('./Chart'));
const science = 'https://github.com/lependens/Capitulo-1/blob/3b4c720627e131bf79163af325b50deaa4b43b4c/scientific_pipeline/docs/ET0_V2_MEMORIA_CIENTIFICA.md';
const nav = {
  inicio:['◫','Visión general','Qué se evaluó y cómo se construyó el experimento.'],
  estaciones:['⌖','Estaciones y datos','Ubicación, calidad, cobertura y observaciones diarias.'],
  comparar:['▥','Comparador','Cruza métodos, configuraciones y estaciones con métricas publicadas.'],
  ahc:['≈','Calibración AHC','Coeficientes de DEVELOPMENT, histórico completo y evaluación retrospectiva.'],
  ml:['◈','Machine Learning','Compara las siete variantes y explora las decisiones de DEVELOPMENT.'],
  ann:['⋮','Redes neuronales','Explora capas, parámetros y las reglas de selección A y E.'],
  temporal:['∿','Análisis temporal','Consulta estimaciones diarias, dispersión y errores en intervalos concretos.'],
  metodo:['≡','Metodología','Entradas, selección, modelos congelados y evaluación retrospectiva.'],
  guia:['?','Guía práctica','Abre ejemplos con una selección inicial preparada.'],
  fuentes:['⊙','Fuentes y auditoría','Identidad de la publicación, huellas y límites de interpretación.'],
} as const;
type Page = keyof typeof nav;
const stages=['Datos SIAR','Familia de entradas','Calendario','AHC · ML · ANN','Selección en DEVELOPMENT','Modelos congelados','TEST retrospectivo'];
const variants=['LINEAR','RIDGE','ELASTIC_NET','TREE','RANDOM_FOREST','EXTRA_TREES','HIST_GRADIENT_BOOSTING'];
const examples: [string,Page,Preset?][] = [
  ['AHC anual, estacional y mensual en IB01','comparar',{station:'IB01',methods:['AHC']}],
  ['Los siete algoritmos ML en HG-Rs','ml'],
  ['Arquitectura ANN seleccionada en IB01','ann'],
  ['ANN A y ANN E en todas las estaciones','comparar',{allStations:true,methods:['ANN_A','ANN_E'],calendars:['annual']}],
  ['Errores por estación para un mismo método','comparar',{allStations:true,methods:['AHC'],calendars:['annual']}],
  ['Evolución diaria de ET₀','temporal'],
  ['Cómo interpretar las métricas','metodo'],
];
const counts = [
  ['12','Estaciones'],['9','Estaciones con TEST'],['3','Familias de entradas'],['3','Configuraciones temporales'],
  ['108','Celdas potenciales'],['108','Celdas AHC'],['105','Celdas ML'],['105','Celdas ANN A'],['105','Celdas ANN E'],
  ['4.935','Checkpoints'],['145.701','Ajustes ML y ANN acreditados'],['600','Coeficientes AHC'],
  ['735','Estimadores ML finales'],['630','Redes finales'],['210','Ensembles ANN'],['9.612','Filas de métricas'],['2.081.376','Registros expandidos'],
];
function Table({columns,rows}:{columns:string[];rows:(string|number)[][]}) {
  return <div className="v2-table" role="region" aria-label={columns.join(', ')} tabIndex={0}><table><thead><tr>{columns.map(c=><th key={c} scope="col">{c}</th>)}</tr></thead>
    <tbody>{rows.map((r,i)=><tr key={i}>{r.map((x,j)=>j===0?<th key={j} scope="row">{x}</th>:<td key={j}>{x}</td>)}</tr>)}</tbody></table></div>;
}
function ScienceLink({anchor,label='Abrir capítulo de la memoria'}:{anchor:string;label?:string}) {
  return <a href={science+'#'+anchor}>{label} ↗</a>;
}
function Methodology(){
  return <section className="p2-panel"><ol className="p2-flow">{stages.map(x=><li key={x}>{x}</li>)}</ol>
    <h2>Leer métricas y calendarios</h2>
    <Table columns={['Indicador','Interpretación']} rows={[
      ['RMSE','Error en mm/día que da mayor peso a diferencias grandes.'],
      ['MAE','Error absoluto medio en mm/día.'],
      ['Sesgo','Media con signo de estimación menos PM en mm/día.'],
      ['R²','Concordancia predictiva relativa. Puede ser negativo.'],
      ['Cobertura','Días con predicción respecto al calendario.'],
      ['n','Pares que entraron en la métrica.'],
    ]}/>
    <p>Anual, estacional y mensual describen la configuración del ajuste o las variables de calendario. Todas las estimaciones son diarias. DEVELOPMENT se empleó en el ajuste y en la selección. La evaluación retrospectiva 2020–2025 usa modelos congelados.</p>
    <ScienceLink anchor="6-diseño-experimental"/></section>;
}
function Development({catalog,page}:{catalog:Catalog;page:'ahc'|'ml'|'ann'}){
  const [station,setStation]=useState('IB01');
  const [family,setFamily]=useState('HG_RS_V1');
  const [calendar,setCalendar]=useState('annual');
  const [variant,setVariant]=useState('ANN_A');
  const [arch,setArch]=useState('64');
  const dimension=(catalog.formulas[family]?.inputs.length??2)+({annual:0,seasonal:4,monthly:12}[calendar]??0);
  const layers=[dimension,...arch.split('-').map(Number),1];
  const parameterCount=layers.slice(1).reduce((sum,q,index)=>sum+(layers[index]+1)*q,0);
  const example=station==='IB01'&&family==='HG_RS_V1';
  const history=useRead<{rows:{coefficient:number;development_pairs:number;retrospective_pairs:number;rmse:number}[]}>(page==='ahc'?'in-sample?'+new URLSearchParams({station,formula:family}):'');
  const historic=history?.data?.rows[0];
  const ahc=memo.ahc.filter(row=>row[0]===calendarNames[calendar]);
  const testKey=catalog.group_keys.station?.includes(station)?station:null;
  const methods=page==='ahc'?['AHC','FORMULA']:page==='ml'?variants.map(v=>'ML_'+v):['ANN_A','ANN_E'];
  const testQuery=new URLSearchParams({formula:family,calendar,group:'station',key:testKey??''});
  methods.forEach(method=>testQuery.append('method',method));
  const test=useRead<{rows:{method:string;n:number;rmse:number;mae:number;bias:number;r2:number}[]}>(testKey?'metrics?'+testQuery:'');
  return <>
    <section className="p2-panel"><div className="p2-filters">
      <Select label="Estación" options={catalog.stations} value={station} onChange={setStation}/>
      <Select label="Familia de entradas" options={Object.fromEntries(Object.entries(catalog.formulas).map(([k,v])=>[k,v.label]))} value={family} onChange={setFamily}/>
      <Select label="Configuración del modelo" options={calendarNames} value={calendar} onChange={setCalendar}/>
    </div><p>{catalog.formulas[family]?.inputs.join(' · ')}</p></section>
    <section className="p2-panel">
      <h2>DEVELOPMENT · ajuste y selección</h2>
      {page==='ahc'&&<>
        <p className="p2-equation">ET₀ ajustada = c × ET₀ de la fórmula</p>
        <p>Por mínimos cuadrados sin intercepto, c = Σ(x × PM) / Σ(x²), para pares elegibles del grupo y con denominador positivo.</p>
        {example?<><p>IB01 · HG-Rs. Ajuste entre el 30-03-2004 y el 31-12-2019.</p>
          <Suspense fallback={<p>Cargando gráfico…</p>}><Chart label="Coeficientes AHC en DEVELOPMENT" option={{
            tooltip:{trigger:'axis'},xAxis:{type:'category',data:ahc.map(r=>r[1])},
            yAxis:{type:'value',name:'Coeficiente c'},series:[{type:'bar',name:'AHC',itemStyle:{color:methodColor('AHC')},data:ahc.map(r=>Number(r[5].replace(',','.')))}],
          }}/></Suspense><Table columns={['Configuración','Grupo','Meses','Pares','Σx²','Coeficiente']} rows={ahc}/></>:
          <p>La memoria incluye la ficha completa de IB01 / HG-Rs. Para esta celda falta la extracción del freeze DEVELOPMENT.</p>}
        <ScienceLink anchor="72-ib01-y-hg-rs-desde-el-dato-hasta-el-error"/>
      </>}
      {page==='ml'&&<>
        <p>Se probaron 37 candidatos por celda. Cada algoritmo aporta su configuración congelada. El principal se eligió entre esas siete configuraciones durante DEVELOPMENT.</p>
        <Table columns={['Algoritmo','Candidatos','Método']} rows={variants.map((v,i)=>[methodName('ML_'+v),[1,6,12,6,4,4,4][i],['Mínimos cuadrados','Penalización L2','Penalizaciones L1 y L2','Umbrales recursivos','Promedio de árboles con bootstrap','Umbrales aleatorios sin bootstrap','Correctores sucesivos mediante árboles'][i]])}/>
        {example&&calendar==='annual'?<><h3>Configuraciones elegidas · IB01 / HG-Rs anual</h3><p>ML principal: Hist Gradient Boosting.</p><Table columns={['Algoritmo','Hiperparámetros','Folds','Media RMSE','DE RMSE','ID abreviado']} rows={memo.ml}/></>:
          <p>La ficha de configuraciones seleccionadas está disponible para IB01 / HG-Rs anual. La matriz completa requiere extraer los índices y fichas congelados.</p>}
        <ScienceLink anchor="k-malla-ml-completa" label="Malla completa de hiperparámetros"/>
      </>}
      {page==='ann'&&<>
        <div className="p2-filters">
          <Select label="Regla ANN" value={variant} onChange={setVariant} options={{ANN_A:'A · último 15 % de fechas',ANN_E:'E · 15 % por orden de hash'}}/>
          <Select label="Arquitectura candidata" value={arch} onChange={setArch} options={Object.fromEntries(['16','32','64','64-32','64-32-16'].map(a=>[a,a]))}/>
        </div>
        <ol className="p2-flow" aria-label="Capas densas conectadas">{layers.map((n,i)=><li key={i}><strong>{n}</strong> {i===0?'entradas':i===layers.length-1?'salida lineal':'unidades ReLU'}</li>)}</ol>
        <p>{parameterCount.toLocaleString('es-ES')} parámetros por red · tres semillas 20261101, 20261102 y 20261103 · ensemble de tres estimaciones.</p>
        {example&&calendar==='annual'?<><h3>Selección documentada · IB01 / HG-Rs anual</h3><p>Seleccionada: 64 unidades. Épocas finales por semilla: {variant==='ANN_A'?'199 / 188 / 191':'199 / 194 / 198'}.</p>
          <Table columns={['Regla','Arquitectura','Folds','Media RMSE','DE RMSE','Seleccionada']} rows={memo.ann.filter(row=>row[0]===variant)}/></>:
          <p>Las dimensiones del esquema corresponden a la selección. El detalle de candidatos y épocas está disponible en la ficha documentada IB01 / HG-Rs anual.</p>}
        <details><summary>Frecuencia de selección por arquitectura</summary><Table columns={['Regla','Arquitectura','Total','HS-Ra','HG-Rs','Valiantzas','Anual','Estacional','Mensual']} rows={memo.annFrequency.filter(row=>row[0]===variant)}/></details>
        <ScienceLink anchor="9-redes-ann-a-y-ann-e"/>
      </>}
    </section>
    {page==='ahc'&&<section className="p2-panel"><h2>Histórico completo · dentro de muestra</h2>
      {historic?<p>Coeficiente anual {fmt(historic.coefficient)} · pares de DEVELOPMENT {historic.development_pairs} · pares retrospectivos {historic.retrospective_pairs} · RMSE {fmt(historic.rmse)} mm/día.</p>:<p>{history?.error??(history?.data?'Sin fila de histórico completo.':'Consultando histórico…')}</p>}
      <p>El ajuste incluye los días evaluados en esta vista.</p></section>}
    <section className="p2-panel"><h2>TEST retrospectivo · 2020–2025</h2>
      {page==='ann'&&<p>Estas métricas corresponden a los ensembles seleccionados y congelados. El selector de arquitectura explora candidatos de DEVELOPMENT.</p>}
      {test?.data?.rows.some(r=>r.n>0)?<Table columns={['Método','Pares','RMSE','MAE','Sesgo','R²']} rows={test.data.rows.map(r=>[methodName(r.method),r.n,fmt(r.rmse),fmt(r.mae),fmt(r.bias),fmt(r.r2)])}/>:
        <p role="status">{test?.error??(!testKey||test?.data?'Esta estación no tiene resultados TEST para esta selección.':'Consultando resultados…')}</p>}
    </section>
  </>;
}
function Temporal({catalog,station}:{catalog:Catalog;station:string}){
  return <><section className="p2-panel"><details><summary>Configuración del modelo y agrupación de resultados</summary><p>La configuración define el calendario utilizado al ajustar el modelo. La agrupación resume estimaciones diarias en el periodo elegido. Las fechas libres afectan solo a las series y a la dispersión.</p></details></section>
    <Workspace catalog={catalog} initialStation={station} initialView="series"/></>;
}
function currentPage():Page {
  const key=new URLSearchParams(location.search).get('seccion')??'inicio';
  return Object.hasOwn(nav,key)?key as Page:'inicio';
}
export default function Et0V2Phase2(){
  const lookup=useRead<Catalog>('catalog');
  const [page,setPage]=useState<Page>(currentPage);
  const [station,setStation]=useState('IB01');
  const [preset,setPreset]=useState<Preset>({});
  const [viewKey,refresh]=useState(0);
  const [expanded,setExpanded]=useState(false);
  const catalog=lookup?.data;
  useEffect(()=>{
    const listener=()=>{setPage(currentPage());setPreset({});refresh(v=>v+1);};
    window.addEventListener('popstate',listener);
    return ()=>window.removeEventListener('popstate',listener);
  },[]);
  function go(next:Page,selection:Preset={}){
    setPage(next);setPreset(selection);refresh(v=>v+1);setExpanded(false);
    const url=new URL(location.href);url.searchParams.set('seccion',next);history.pushState(null,'',url);
    window.scrollTo(0,0);
  }
  if(!catalog)return <main className="p2-loading" role="status">{lookup?.error??'Consultando el catálogo ET₀…'}</main>;
  return <div className="p2 v2-preview">
    <a className="p2-skip" href="#contenido-et0">Ir al contenido</a>
    <aside className="p2-sidebar">
      <a className="p2-brand" href="?science=et0-v2">SIAR <strong>ET₀ V2</strong></a>
      <span>Baleares · explorador científico</span>
      <button className="p2-menu" aria-expanded={expanded} onClick={()=>setExpanded(v=>!v)}>Menú de análisis</button>
      <nav className={expanded?'p2-open':''} aria-label="Secciones del explorador">{Object.entries(nav).map(([id,[icon,title]])=>
        <button key={id} aria-current={page===id?'page':undefined} onClick={()=>go(id as Page)}><span aria-hidden="true">{icon}</span>{title}</button>
      )}</nav>
    </aside>
    <main id="contenido-et0" className="p2-main">
      <header className="p2-header"><p className="p2-status">{catalog.status==='VERIFIED'?'Resultados verificados':catalog.status} · {catalog.publication_status==='NOT_PUBLISHED'?'Pendientes de publicación científica':catalog.publication_status}</p>
        <h1>{nav[page][1]}</h1><p>{nav[page][2]}</p></header>
      {page==='inicio'&&<>
        <section className="p2-panel"><h2>Del dato diario a la evaluación</h2><ol className="p2-flow">{stages.map(x=><li key={x}>{x}</li>)}</ol><p>Compara estimaciones de ET₀ con PM FAO56 calculada durante la retrospectiva 2020–2025.</p>
          <button onClick={()=>go('comparar')}>Abrir comparador</button>{' '}<button onClick={()=>go('guia')}>Explorar ejemplos</button></section>
        <section className="p2-panel"><h2>Inventario experimental</h2><dl className="p2-counts">{counts.map(([n,l])=><div key={l}><dt>{l}</dt><dd>{n}</dd></div>)}</dl>
          <p>Cada celda combina estación, familia de entradas y configuración temporal. Un candidato es una arquitectura o combinación de hiperparámetros. Cada ajuste entrena una instancia. El checkpoint conserva evidencia de un candidato. Los modelos finales se reajustan sobre DEVELOPMENT y se congelan.</p><ScienceLink anchor="101-inventario-auditado-de-ajustes-y-artefactos"/></section>
      </>}
      {page==='estaciones'&&<><section className="p2-panel"><Select label="Ir a estación" value={station} options={Object.fromEntries(Object.entries(catalog.stations).map(([k,v])=>[k,k+' · '+v]))} onChange={setStation}/></section><StationPage key={station} catalog={catalog} current={station} select={setStation} compare={()=>go('comparar',catalog.station_details?.some(s=>s.code===station&&s.retrospective)?{station}:{})}/></>}
      {page==='comparar'&&<Compare key={viewKey} catalog={catalog} preset={preset}/>}
      {(page==='ahc'||page==='ml'||page==='ann')&&<><Development key={page} catalog={catalog} page={page}/><details className="p2-panel"><summary>Comparar estaciones y configuraciones</summary><Compare key={viewKey} catalog={catalog} preset={{methods:page==='ml'?variants.map(v=>'ML_'+v):page==='ann'?['ANN_A','ANN_E']:['AHC']}}/></details></>}
      {page==='temporal'&&<Temporal catalog={catalog} station={station}/>}
      {page==='metodo'&&<Methodology/>}
      {page==='guia'&&<section className="p2-panel"><ol className="p2-examples">{examples.map(([label,next,selection])=><li key={label}><button onClick={()=>go(next,selection)}>{label} →</button></li>)}</ol><p>Usa Restablecer filtros para recuperar la selección del ejemplo.</p></section>}
      {page==='fuentes'&&<section className="p2-panel"><Table columns={['Identidad','Valor']} rows={[
        ['Versión',catalog.version],['Referencia',catalog.reference],['Estado técnico',catalog.status??'—'],['Publicación científica',catalog.publication_status??'—'],
        ['Catálogo SHA-256',catalog.catalog_sha256??'—'],['Freeze SHA-256',catalog.freeze_sha256??'—'],['Dataset SHA-256',catalog.dataset_sha256??'—'],
        ['Catálogo de estaciones SHA-256',catalog.station_catalog_sha256??'—'],['Manifiesto original SHA-256',catalog.original_manifest_sha256??'—'],['Ejecución',catalog.source_run_id??'—'],
      ]}/><h2>Límites de la evidencia</h2><p>PM FAO56 es calculada. El AHC de histórico completo se evalúa dentro de muestra. Se conservan las predicciones negativas. IB07, IB09 e IB101 carecen de TEST retrospectivo. La evaluación disponible mide concordancia retrospectiva y no acredita generalización prospectiva o espacial.</p>
      <details><summary>Limitaciones del catálogo firmado</summary><ul>{catalog.limitations?.map(x=><li key={x}>{x}</li>)}</ul></details>
      <ScienceLink anchor="g-nota-dimensional-y-límites-de-interpretación-de-valiantzasmartí" label="Convención dimensional de Valiantzas–Martí"/><br/>
      <ScienceLink anchor="m-posibilidades-de-explotación-y-web" label="Inventario de datos disponibles"/></section>}
    </main>
  </div>;
}
