import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  base: '/Sign-Translate/',
  plugins: [react()],
  resolve: {
    conditions: ['module', 'browser', 'development|production', 'onnxruntime-web-use-extern-wasm'],
  },
})
