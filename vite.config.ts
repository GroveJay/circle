import { defineConfig } from 'vite'
import react, { reactCompilerPreset } from '@vitejs/plugin-react'
import basicSsl from '@vitejs/plugin-basic-ssl'
import babel from '@rolldown/plugin-babel'

// https://vite.dev/config/
export default defineConfig({
  base: "/circle/",
  plugins: [
    react(),
    basicSsl(),
    babel({ presets: [reactCompilerPreset()] })
  ],
})
