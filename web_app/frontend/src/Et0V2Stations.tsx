import {lazy, Suspense, useState} from 'react';
import {Select, useRead} from './Et0V2Preview';
import type {Catalog, Station} from './Et0V2Preview';
import coast from './balearic-coast.json';

const Chart = lazy(() => import('./Chart'));
const variables = {
  TempMedia:['Temperatura media','°C'], TempMin:['Temperatura mínima','°C'], TempMax:['Temperatura máxima','°C'],
  HumedadMedia:['Humedad relativa media','%'], VelViento:['Velocidad del viento','m/s'],
  Radiacion:['Radiación','MJ/m²/día'], Precipitacion:['Precipitación','mm/día'],
  ET0_PM_FAO56_V1:['ET₀ PM FAO56 calculada','mm/día'],
} as const;
type Variable = keyof typeof variables;
type Row = Record<string, string | number | null>;
type Weather = {points:Row[]; overall:Row; stride:number; paired_count?:number};
const labels = new Intl.DateTimeFormat('es-ES',{month:'short',timeZone:'UTC'});
const number = new Intl.NumberFormat('es-ES',{maximumFractionDigits:2});
const colors=['#076d68','#8d488f','#ae4e22','#2e58a2','#84400d','#548330','#b52154','#304556'];
const value = (row:Row|undefined,key:Variable) => typeof row?.[key]==='number' ? row[key] as number:null;
const show = (v:number|null) => v===null?'—':number.format(v);

function StationMap({stations,current,select}:{stations:Station[];current:string;select:(code:string)=>void}) {
  const located=stations.filter(s=>s.latitude!=null&&s.longitude!=null);
  const [west,south,east,north]=coast.bounds;
  if (!located.length) return <p>El catálogo de esta instancia no incluye coordenadas verificadas.</p>;
  return <div className="v2-map v2-geographic" role="group" aria-label="Mapa georreferenciado de las estaciones SIAR">
    <svg viewBox="0 0 1000 600" preserveAspectRatio="none" aria-hidden="true">
      <path d={coast.path} fill="#d2e1cf" stroke="#628574" strokeWidth="1.25" fillRule="evenodd"/>
    </svg>
    {located.map(s=><button key={s.code} className="v2-marker"
      style={{left:`${(s.longitude!-west)/(east-west)*100}%`,top:`${(north-s.latitude!)/(north-south)*100}%`}}
      aria-label={s.code+' · '+s.name} aria-pressed={current===s.code}
      title={s.code+' · '+s.name} onClick={()=>select(s.code)}>{s.code.slice(2)}</button>)}
    <span className="v2-map-west">Ibiza y Formentera</span><span className="v2-map-center">Mallorca</span><span className="v2-map-east">Menorca</span>
  </div>;
}

export function StationPage({catalog,current,select,compare}:{
  catalog:Catalog;current:string;select:(code:string)=>void;compare:()=>void;
}) {
  const [selected,setSelected]=useState<Variable[]>(['TempMedia']);
  const [second,setSecond]=useState<Variable|''>('');
  const [view,setView]=useState('daily');
  const [stationMean,setStationMean]=useState(true);
  const [start,setStart]=useState('');
  const [end,setEnd]=useState('');
  const [offset,setOffset]=useState(0);
  const station=catalog.station_details?.find(s=>s.code===current);
  const summary=useRead<{variables:Variable[];rows:Row[]}>('weather/stations');
  const options=summary?.data?.variables.filter(key=>key in variables)??[];
  const active=selected.filter(key=>options.includes(key));
  const first=active[0];
  const scatter=!!first&&!!second&&first!==second;
  const mode=scatter?'scatter':view;
  const valid=!start||!end||start<=end;
  const query=new URLSearchParams({station:current,view:mode});
  if(start)query.set('start',start);
  if(end)query.set('end',end);
  const keys=scatter&&second?[first,second]:active;
  keys.forEach(key=>query.append('variable',key));
  const response=useRead<Weather>(valid&&keys.length?'weather/series?'+query:'');
  const points=response?.data?.points??[];
  const dailyQuery=new URLSearchParams({station:current,offset:String(offset),limit:'50'});
  if (start) dailyQuery.set('from',start);
  if (end) dailyQuery.set('to',end);
  const [showDaily,setShowDaily]=useState(false);
  const daily=useRead<{rows:Row[]}>(showDaily&&valid&&keys.length?'/api/observations?'+dailyQuery:'');
  const choose=(code:string)=>{setStart('');setEnd('');setOffset(0);select(code);};
  const keyLabel=(key:Variable)=>variables[key][0]+' ('+variables[key][1]+')';
  const periodLabel=(period:string|number|null)=>mode==='month'?labels.format(new Date(Date.UTC(2020,Number(period)-1,1))):String(period??'—');
  const groups=[...new Set(keys.map(k=>variables[k][1]))];
  const plots=groups.length>2?groups.map(unit=>keys.filter(key=>variables[key][1]===unit)):[keys];
  const history=summary?.data?.rows.find(row=>row.station_code===current);
  return <>
    <section className="v2-panel"><h2>Estaciones SIAR en Baleares</h2>
      <StationMap stations={catalog.station_details??[]} current={current} select={choose}/>
      {catalog.station_catalog_sha256&&<p className="v2-subtle">Posiciones del catálogo SIAR. Costa de Natural Earth 1:10 millones. Selecciona una estación en el mapa o en la tabla.</p>}
      <div className="v2-table" role="region" aria-label="Estaciones y promedios históricos" tabIndex={0}>
        <table><thead><tr><th>Estación</th><th>Periodo de catálogo</th><th>Quality V1</th><th>Retrospectiva V2</th>
          {options.map(key=><th scope="col" key={key}>{keyLabel(key)}</th>)}</tr></thead>
          <tbody>{catalog.station_details?.map(s=>{
            const row=summary?.data?.rows.find(x=>x.station_code===s.code);
            return <tr key={s.code}><th scope="row"><button className="v2-link" onClick={()=>choose(s.code)}>{s.code} · {s.name}</button></th>
              <td>{s.official_from} – {s.official_to}</td><td>{s.quality_status}</td><td>{s.retrospective?'Disponible':'Serie histórica'}</td>
              {options.map(key=><td key={key}>{show(value(row,key))}<small className="p2-cell-note">{row?.[key+'_n']??0} días</small></td>)}</tr>;
          })}</tbody></table>
      </div>
      <p>Medias de los valores diarios disponibles en cada estación. Los periodos y días válidos pueden variar entre variables.</p>
      {summary?.error&&<p role="status">No se pudieron consultar las medias por estación.</p>}
    </section>
    <section className="v2-panel"><h2>{current} · {catalog.stations[current]}</h2>
      <div className="v2-actions"><button disabled={station?.retrospective===false} onClick={compare}>Comparar modelos en esta estación</button>
        <button onClick={()=>window.print()}>Imprimir ficha</button></div>
      {station&&<p>{station.municipality}{station.altitude!=null?' · '+station.altitude+' m':''} · Cobertura SIAR {station.coverage_percent}% · Quality V1 {station.quality_status}.</p>}
      {station?.retrospective===false&&<p>Estación histórica sin TEST retrospectivo de ET₀ V2.</p>}
      {current==='IB09'&&<p>IB09 conserva Quality V1 FAIL. Su admisión excepcional se limita a DEVELOPMENT.</p>}
      <fieldset><legend>Variables meteorológicas</legend><div className="p2-checks">{options.map(key=><label key={key}><input type="checkbox" checked={active.includes(key)}
        onChange={e=>{if(key===second)setSecond('');setSelected(e.target.checked?[...selected,key]:selected.filter(x=>x!==key));}}/>{keyLabel(key)}</label>)}</div></fieldset>
      <div className="v2-filters">
        <Select label="Vista histórica" value={view} options={{daily:'Serie cronológica',year:'Medias por año',month:'Medias por mes del calendario'}}
          onChange={v=>{setView(v);setSecond('');}}/>
        <Select label="Segunda variable · dispersión" value={second} options={{'':'Sin dispersión',...Object.fromEntries(options.filter(k=>k!==first).map(k=>[k,keyLabel(k)]))}}
          onChange={id=>setSecond(id as Variable|'')}/>
        <label>Desde<input aria-label="Desde" type="date" value={start} onChange={e=>{setOffset(0);setStart(e.target.value);}}/></label>
        <label>Hasta<input aria-label="Hasta" type="date" value={end} onChange={e=>{setOffset(0);setEnd(e.target.value);}}/></label>
      </div>
      {history&&<p>Datos disponibles del {history.first_date} al {history.last_date}. Deja las fechas vacías para consultar todo el histórico.</p>}
      {!scatter&&(mode==='year'||mode==='month')&&<label className="p2-inline"><input type="checkbox" checked={stationMean} onChange={e=>setStationMean(e.target.checked)}/> Mostrar media histórica de la estación</label>}
      {scatter&&<p>Cada punto muestra un día con las dos observaciones disponibles. El eje horizontal utiliza la primera variable marcada. La nube describe asociación entre variables.</p>}
      {response?.data&&response.data.stride>1&&<p>{scatter?`Muestreo visual de un par de cada ${response.data.stride}.`:`Vista cronológica agrupada en bloques de hasta ${response.data.stride} días con observación. Se muestran las medias de cada bloque.`}</p>}
      {mode==='month'&&<p>Cada mes agrupa todos los días disponibles de ese mes dentro de las fechas elegidas.</p>}
      {points.length>0?<Suspense fallback={<p>Cargando gráfico…</p>}>
        {scatter&&second?<Chart label={keyLabel(first)+' frente a '+keyLabel(second)} option={{
          tooltip:{trigger:'item'},grid:{left:62,right:32,bottom:60},xAxis:{type:'value',name:keyLabel(first),nameLocation:'middle',nameGap:30},
          yAxis:{type:'value',name:variables[second][1]},series:[{type:'scatter',symbolSize:5,data:points.map(row=>[row[first],row[second],row.date])}],
        }}/>:plots.map(vars=>{
          const units=[...new Set(vars.map(key=>variables[key][1]))];
          return <Chart key={vars.join('/')} label={vars.map(keyLabel).join(' · ')} option={{
            tooltip:{trigger:'axis'},legend:{type:'scroll',bottom:0},grid:{left:62,right:units.length>1?60:24,bottom:60},
            xAxis:{type:mode==='daily'?'time':'category',data:mode==='daily'?undefined:points.map(p=>periodLabel(p.period))},
            yAxis:units.map(unit=>({type:'value',name:unit,scale:true})),animation:false,
            series:vars.flatMap(key=>{
              const axis=units.indexOf(variables[key][1]);
              const color=colors[Object.keys(variables).indexOf(key)];
              const overall=value(history,key);
              const series={name:keyLabel(key),type:mode==='daily'?'line':'bar',yAxisIndex:axis,itemStyle:{color},showSymbol:false,
                data:points.map(row=>mode==='daily'?[row.period,value(row,key)]:value(row,key))};
              return mode!=='daily'&&stationMean&&overall!==null?[series,{name:'Media histórica · '+variables[key][0],type:'line',yAxisIndex:axis,itemStyle:{color},
                lineStyle:{type:'dashed'},showSymbol:false,data:points.map(()=>overall)}]:[series];
            }),
          }}/>;
        })}
      </Suspense>:<p role="status">{!valid?'La fecha inicial debe ser anterior o igual a la final.':!keys.length?'Selecciona al menos una variable.':response?.error??(response?.data?'Sin observaciones para esta selección.':'Consultando observaciones…')}</p>}
      {points.length>0&&(mode==='year'||mode==='month')&&
        <details><summary>Medias y número de días por periodo</summary><div className="v2-table" tabIndex={0}><table><thead><tr><th>Periodo</th>{keys.map(key=><th key={key}>{keyLabel(key)}</th>)}</tr></thead>
          <tbody>{points.map(row=><tr key={row.period}><th scope="row">{periodLabel(row.period)}</th>{keys.map(key=><td key={key}>{show(value(row,key))}<small className="p2-cell-note">{row[key+'_n']} días</small></td>)}</tr>)}</tbody></table></div></details>}
      <details onToggle={event=>setShowDaily(event.currentTarget.open)}><summary>Observaciones diarias en tabla</summary>
        {!!daily?.data?.rows.length&&<div className="v2-table" role="region" aria-label="Observaciones diarias" tabIndex={0}><table>
          <thead><tr><th>Fecha</th>{keys.map(key=><th key={key}>{keyLabel(key)}</th>)}</tr></thead>
          <tbody>{daily.data.rows.map(row=><tr key={row.date}><th scope="row">{row.date}</th>{keys.map(key=><td key={key}>{row[key]||'—'}</td>)}</tr>)}</tbody>
        </table></div>}
        <div className="v2-actions"><button disabled={!offset} onClick={()=>setOffset(offset-50)}>Página anterior</button>
          <button disabled={(daily?.data?.rows.length??0)<50} onClick={()=>setOffset(offset+50)}>Página siguiente</button></div>
      </details>
      <p className="v2-subtle">Fuente: observaciones del Dataset V1. Agregaciones descriptivas sobre lecturas disponibles de cada variable.</p>
    </section>
  </>;
}
