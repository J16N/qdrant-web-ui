import { defineConfig } from 'vite';
import reactRefresh from '@vitejs/plugin-react';
import svgrPlugin from 'vite-plugin-svgr';
import eslintPlugin from 'vite-plugin-eslint';
import path from 'path';
import { rehypeMetaAsAttributes } from "./src/lib/rehype-meta-as-attributes";
import wasm from "vite-plugin-wasm";
import topLevelAwait from "vite-plugin-top-level-await";

// this is required for wasm
const viteServerConfig = {
  name: "log-request-middleware",
  configureServer(server) {
    server.middlewares.use((req, res, next) => {
      res.setHeader("Access-Control-Allow-Origin", "*");
      res.setHeader("Access-Control-Allow-Methods", "GET");
      res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
      res.setHeader("Cross-Origin-Embedder-Policy", "require-corp");
      next();
    });
  }
};

// https://vitejs.dev/config/
export default defineConfig(async () => {
  const mdx = await import('@mdx-js/rollup');

  return {
    base: './',
    // This changes the output dir from dist to build
    // comment this out if that isn't relevant for your project
    build: {
      outDir: 'dist',
    },
    optimizeDeps: {
      exclude: ['wasm-dist-bhtsne'],
    },
    plugins: [
      viteServerConfig,
      wasm(),
      topLevelAwait(),
      reactRefresh(),
      svgrPlugin({
        svgrOptions: {
          icon: true,
          // ...svgr options (https://react-svgr.com/docs/options/)
        },
      }),
      eslintPlugin({
        include: ['src/**/*.jsx', 'src/**/*.js', 'src/**/*.ts', 'src/**/*.tsx'],
        exclude: [
          'node_modules/**',
          'dist/**, build/**',
          '**/*.mdx',
          '**/*.md'],
      }),
      mdx.default({
        rehypePlugins: [
          rehypeMetaAsAttributes,
        ],
      }),
    ],
    test: {
      globals: true,
      environment: 'jsdom',
      setupFiles: ['./src/setupTests.js'],
    },
    resolve: {
      alias: {
        "wasm_dist_bhtsne": path.resolve(__dirname, "./wasm_dist_bhtsne/pkg"),
        "wasm-tsne": path.resolve(__dirname, "./wasm-tsne/pkg"),
        "wasm-bhtsne": path.resolve(__dirname, "./wasm-bhtsne/pkg"),
        "examples": path.resolve(__dirname, "./examples"),
      }
    }
  }
});
