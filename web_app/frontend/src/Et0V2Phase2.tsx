import {useEffect, useState} from 'react';
import {StationPage} from './Et0V2Stations';
import {Select, useRead, Workspace} from './Et0V2Preview';
import type {Catalog} from './Et0V2Preview';
import {Compare} from './Et0V2Compare';
import {Development, Table, variants} from './Et0V2Development';
import type {DevelopmentCatalog} from './Et0V2Development';
import type {Preset} from './Et0V2Compare';
import './et0-phase2.css';

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
const examples: [string,Page,string,Preset?][] = [
  ['AHC anual, estacional y mensual en IB01','comparar','Selecciona AHC anual como referencia. Compara RMSE y sesgo con los ajustes estacional y mensual, y comprueba n y la huella.',{station:'IB01',methods:['AHC']}],
  ['Los siete algoritmos ML en HG-Rs','ml','Filtra los candidatos en DEVELOPMENT por algoritmo. Consulta sus folds y, en TEST, los siete estimadores congelados.'],
  ['Arquitectura ANN seleccionada en IB01','ann','Alterna ANN A y ANN E para ver las capas, la arquitectura seleccionada y las épocas finales.'],
  ['ANN A y ANN E en todas las estaciones','comparar','Agrupa por estación y comprueba si las dos variantes usan los mismos días.',{allStations:true,methods:['ANN_A','ANN_E'],calendars:['annual']}],
  ['Errores por estación para un mismo método','comparar','Ordena por RMSE. Revisa cobertura y n antes de interpretar una diferencia entre estaciones.',{allStations:true,methods:['AHC'],calendars:['annual']}],
  ['Evolución diaria de ET₀','temporal','Elige una estación y un intervalo. Cambia de la serie diaria a dispersión o residuos.'],
  ['Cómo interpretar las métricas','metodo','Consulta las fórmulas y el significado de las métricas.'],
  ['Enfrentar 2020 y 2025 con tres familias','comparar','Compara el AHC anual en ambas anualidades. Cada año tiene días distintos, así que observa la variación descriptiva.',{station:'IB01',methods:['AHC'],calendars:['annual'],families:['HS_RA_V1','HG_RS_V1','VALIANTZAS_V1'],years:['2020','2025']}],
  ['Medias históricas de una estación','estaciones','Selecciona la estación en el mapa, marca las variables y cambia a Medias por año o Medias por mes del calendario.'],
];
const guidance: [Page,string][] = [
  ['inicio','Sigue el flujo desde los datos SIAR hasta TEST y consulta el tamaño del experimento.'],
  ['estaciones','Pulsa un punto del mapa o una fila. La tabla muestra medias de todo el histórico y días válidos. En la ficha puedes combinar variables, acotar fechas y consultar curvas, medias o dispersión.'],
  ['comparar','Marca familias, años, calendarios, estaciones y métodos. Sin años se muestra 2020–2025 en conjunto. Elige la métrica, ordena la tabla o exporta CSV. Los deltas requieren una cohorte idéntica.'],
  ['ahc','Selecciona estación y familia. Compara coeficientes en el gráfico por mes de aplicación. Abre los folds para ver el ajuste por año excluido. El histórico completo tiene su propio panel dentro de muestra.'],
  ['ml','Elige una celda, examina los 37 candidatos y mira qué configuración congeló cada algoritmo. Abre las frecuencias para saber cuántas celdas eligieron cada principal.'],
  ['ann','Elige A o E, inspecciona cada arquitectura y cambia entre métricas de ensemble y de semilla. La ficha congelada muestra la arquitectura y sus épocas finales.'],
  ['temporal','Selecciona estación, familia y modelos. El calendario pertenece al ajuste y es independiente del intervalo de visualización.'],
  ['metodo','Consulta las ecuaciones de las tres familias, AHC y métricas de error.'],
  ['fuentes','Consulta huellas, procedencia y limitaciones de ambos catálogos.'],
];
const counts = [
  ['12','Estaciones'],['9','Estaciones con TEST'],['3','Familias de entradas'],['3','Configuraciones temporales'],
  ['108','Celdas potenciales'],['108','Celdas AHC'],['105','Celdas ML'],['105','Celdas ANN A'],['105','Celdas ANN E'],
  ['4.935','Checkpoints'],['145.701','Ajustes ML y ANN acreditados'],['600','Coeficientes AHC'],
  ['735','Estimadores ML finales'],['630','Redes finales'],['210','Ensembles ANN'],['9.612','Filas de métricas'],['2.081.376','Registros expandidos'],
];
function ScienceLink({anchor,label='Abrir capítulo de la memoria'}:{anchor:string;label?:string}) {
  return <a href={science+'#'+anchor}>{label} ↗</a>;
}
function Methodology(){
  return <>
    <section className="p2-panel"><ol className="p2-flow">{stages.map(x=><li key={x}>{x}</li>)}</ol>
      <h2>Métricas de evaluación</h2>
      <p>Para cada par válido, yᵢ es PM FAO56 calculada, ŷᵢ es la estimación, eᵢ = ŷᵢ − yᵢ y ȳ es la media de las referencias de esa cohorte.</p>
      <Table columns={['Indicador','Ecuación','Interpretación']} rows={[
        ['RMSE','√(Σ eᵢ² / n)','Error en mm/día que da más peso a diferencias grandes.'],
        ['MAE','Σ |eᵢ| / n','Error absoluto medio en mm/día.'],
        ['Sesgo','Σ eᵢ / n','Signo positivo: sobreestimación. Signo negativo: infraestimación.'],
        ['R²','1 − Σ eᵢ² / Σ (yᵢ − ȳ)²','Concordancia predictiva relativa. Puede ser negativo y requiere variación en la referencia.'],
        ['Cobertura','100 × días con predicción / días de calendario','Disponibilidad en el periodo, expresada en porcentaje.'],
        ['n','Número de pares válidos','Soporte de las métricas de error.'],
      ]}/>
    </section>
    <section className="p2-panel"><h2>Fórmulas empíricas y calibración AHC</h2>
      <p>T = (Tmax + Tmin) / 2 y D = Tmax − Tmin. Ra es la radiación extraterrestre calculada y Rs la radiación solar medida, ambas en MJ/m²/día. Las salidas están en mm/día.</p>
      <Table columns={['Familia','Ecuación aplicada']} rows={[
        ['Hargreaves–Samani Ra','ET₀ = 0,0023 × 0,408 × Ra × (T + 17,8) × √D'],
        ['Hargreaves Rs','ET₀ = 0,0135 × 0,408 × Rs × (T + 17,8)'],
        ['Valiantzas–Martí','Rs estimada = 0,338 × Ra × D⁰·³ × (1,001 − HRmedia/100)⁰·². ET₀ = 0,0135 × 0,408 × Rs estimada × (T + 17,8).'],
        ['AHC','ET₀ ajustada = c_g × ET₀ empírica. c_g = Σ (xᵢ yᵢ) / Σ xᵢ², para pares válidos del grupo g y denominador positivo.'],
      ]}/>
      <p>En AHC, xᵢ es la estimación empírica. El grupo g reúne todo el año en el ajuste anual, una estación climática en el estacional (DJF, MAM, JJA o SON) y un mes del calendario en el mensual. Cada coeficiente se ajusta con sus propios pares en DEVELOPMENT. El histórico completo se ajusta y evalúa dentro de muestra. Todas las estimaciones son diarias.</p>
      <p>Valiantzas–Martí usa humedad relativa media en porcentaje y la conversión 0,408 con la convención dimensional documentada.</p>
      <ScienceLink anchor="g-nota-dimensional-y-límites-de-interpretación-de-valiantzasmartí" label="Ecuaciones y límites dimensionales en la memoria"/>
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
  const metadata=useRead<DevelopmentCatalog>(page==='fuentes'?'development/catalog':'');
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
      {(page==='ahc'||page==='ml'||page==='ann')&&<><Development key={page} catalog={catalog} page={page} initialStation={station}/><details className="p2-panel"><summary>Comparar estaciones y configuraciones</summary><Compare key={viewKey} catalog={catalog} preset={{methods:page==='ml'?variants.map(v=>'ML_'+v):page==='ann'?['ANN_A','ANN_E']:['AHC']}}/></details></>}
      {page==='temporal'&&<Temporal catalog={catalog} station={station}/>}
      {page==='metodo'&&<Methodology/>}
      {page==='guia'&&<>
        <section className="p2-panel"><h2>Qué puedes hacer en cada sección</h2>
          <dl className="p2-guide">{guidance.map(([target,description])=><div key={target}><dt><button onClick={()=>go(target)}>{nav[target][1]} →</button></dt><dd>{description}</dd></div>)}</dl>
        </section>
        <section className="p2-panel"><h2>Probar con datos reales</h2><ol className="p2-examples">
          {examples.map(([label,next,hint,selection])=><li key={label}><button onClick={()=>go(next,selection)}>{label} →</button><p>{hint}</p></li>)}
        </ol><p>En el comparador, usa Restablecer filtros para volver a la selección inicial del ejemplo.</p></section>
      </>}
      {page==='fuentes'&&<section className="p2-panel"><Table columns={['Identidad','Valor']} rows={[
        ['Versión',catalog.version],['Referencia',catalog.reference],['Estado técnico',catalog.status??'—'],['Publicación científica',catalog.publication_status??'—'],
        ['Catálogo SHA-256',catalog.catalog_sha256??'—'],['Freeze SHA-256',catalog.freeze_sha256??'—'],['Dataset SHA-256',catalog.dataset_sha256??'—'],
        ['Catálogo de estaciones SHA-256',catalog.station_catalog_sha256??'—'],['Manifiesto original SHA-256',catalog.original_manifest_sha256??'—'],['Ejecución',catalog.source_run_id??'—'],
      ]}/><h2>Límites de la evidencia</h2><p>PM FAO56 es calculada. El AHC de histórico completo se evalúa dentro de muestra. Se conservan las predicciones negativas. IB07, IB09 e IB101 carecen de TEST retrospectivo. La evaluación disponible mide concordancia retrospectiva y no acredita generalización prospectiva o espacial.</p>
      <h2>Catálogo complementario DEVELOPMENT</h2>
      {metadata?.data?<><p>{metadata.data.source.integrity} · {metadata.data.source.publication_status} · SHA‑256 <code className="p2-hash">{metadata.data.source.catalog_sha256}</code></p>
        <Table columns={['Tabla','Registros','SHA-256']} rows={Object.entries(metadata.data.source.tables).map(([name,entry])=>[name,entry.rows,<code className="p2-hash">{entry.sha256}</code>])}/>
        <a href="https://github.com/lependens/Capitulo-1/blob/9d47cb02300b338c501a7d3146ac05469dcb6376/scientific_pipeline/docs/et0_v2_experimental_metadata.md">Contrato de los metadatos experimentales ↗</a></>:
        <p role="status">{metadata?.error??'Consultando procedencia complementaria…'}</p>}
      <details><summary>Limitaciones del catálogo firmado</summary><ul>{catalog.limitations?.map(x=><li key={x}>{x}</li>)}</ul></details>
      <ScienceLink anchor="g-nota-dimensional-y-límites-de-interpretación-de-valiantzasmartí" label="Convención dimensional de Valiantzas–Martí"/><br/>
      <ScienceLink anchor="m-posibilidades-de-explotación-y-web" label="Inventario de datos disponibles"/></section>}
    </main>
  </div>;
}
