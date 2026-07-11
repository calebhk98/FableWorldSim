import { defineConfig } from "vite";

// Thin static-asset build: no framework plugin needed. The dev server proxies
// nothing — the client talks to the API's own base URL (see src/config.ts),
// which defaults to the compose-published http://localhost:8080 but is
// overridable via the VITE_FWS_API_URL env var (`.env.local`, CI, etc.).
export default defineConfig({
  root: ".",
  server: {
    port: 5173,
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
