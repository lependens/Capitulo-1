import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';
export default defineConfig(({mode}) => ({
  plugins:[react()],
  server:{proxy:{'/api':loadEnv(mode,'.','WEB_VITE_').WEB_VITE_API ?? 'http://127.0.0.1:8001'}},
  build:{rollupOptions:{output:{manualChunks:{react:['react','react-dom']}}}},
}));
