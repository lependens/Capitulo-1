import { useState } from 'react';
import { StationPage } from './Et0V2Stations';
import { Select, useRead, Workspace } from './Et0V2Preview';
import type { Catalog } from './Et0V2Preview';

const sections = {
  home:'Inicio', stations:'Estaciones y datos', models:'Modelos y metodología',
  results:'Resultados por familia', compare:'Comparar modelos', sources:'Fuentes y auditoría',
};
type Section = keyof typeof sections;
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
