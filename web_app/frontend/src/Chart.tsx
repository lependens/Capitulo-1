import { useEffect, useRef } from 'react';
import * as echarts from 'echarts/core';
import { LineChart, ScatterChart, BarChart } from 'echarts/charts';
import { GridComponent, TooltipComponent, LegendComponent, DataZoomComponent, ToolboxComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';
import type { EChartsCoreOption } from 'echarts/core';
echarts.use([LineChart,ScatterChart,BarChart,GridComponent,TooltipComponent,LegendComponent,DataZoomComponent,ToolboxComponent,CanvasRenderer]);
export default function Chart({option,label}:{option:EChartsCoreOption;label:string}){const root=useRef<HTMLDivElement>(null);useEffect(()=>{if(!root.current)return;const chart=echarts.init(root.current,undefined,{renderer:'canvas'});chart.setOption(option);const observer=new ResizeObserver(()=>chart.resize());observer.observe(root.current);return()=>{observer.disconnect();chart.dispose();};},[option]);return <div className="chart" role="img" aria-label={label} ref={root}/>;}
