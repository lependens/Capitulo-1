import React from 'react';
import { createRoot } from 'react-dom/client';
import Legacy from './AppV2';
import Explorer from './Explorer';
import './style.css';
const Et0V2Preview = React.lazy(() => import('./Et0V2Preview'));
const Et0V2 = React.lazy(() => import('./Et0V2'));
const Phase2 = React.lazy(() => import('./Et0V2Phase2'));
const params = new URLSearchParams(location.search);
createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    {params.has('legacy') ? <Legacy/> : params.get('science') === 'et0-v2' ?
      <React.Suspense fallback={<p role="status">Cargando explorador…</p>}>{params.has('phase1') ? <Et0V2/> : <Phase2/>}</React.Suspense> :
      params.get('preview') === 'et0-v2' ?
      <React.Suspense fallback={<p role="status">Cargando vista previa…</p>}><Et0V2Preview/></React.Suspense> : <Explorer/>}
  </React.StrictMode>,
);
