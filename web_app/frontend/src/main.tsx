import React from 'react';
import { createRoot } from 'react-dom/client';
import Legacy from './AppV2';
import Explorer from './Explorer';
import './style.css';
const Et0V2Preview = React.lazy(() => import('./Et0V2Preview'));
const params = new URLSearchParams(location.search);
createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    {params.has('legacy') ? <Legacy/> : params.get('preview') === 'et0-v2' ?
      <React.Suspense fallback={<p role="status">Cargando vista previa…</p>}><Et0V2Preview/></React.Suspense> : <Explorer/>}
  </React.StrictMode>,
);
