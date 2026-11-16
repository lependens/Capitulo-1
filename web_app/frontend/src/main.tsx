import React from 'react';
import { createRoot } from 'react-dom/client';
import Legacy from './AppV2';
import Explorer from './Explorer';
import './style.css';
createRoot(document.getElementById('root')!).render(<React.StrictMode>{new URLSearchParams(location.search).has('legacy')?<Legacy/>:<Explorer/>}</React.StrictMode>);
