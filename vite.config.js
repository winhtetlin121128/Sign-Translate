import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  // Relative assets work on both Vercel's domain root and GitHub Pages' repo subpath.
  base: './',
  plugins: [react()],
  resolve: {
    conditions: ['module', 'browser', 'development|production', 'onnxruntime-web-use-extern-wasm'],
  },
})

