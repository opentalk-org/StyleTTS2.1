import { fileURLToPath, URL } from "node:url";

import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const metricsApi = process.env.VITE_METRICS_API_URL ?? "http://127.0.0.1:8182";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) } },
  // The viewer talks to its own origin so a forwarded dev port needs no second
  // tunnel, and the browser never makes a cross-origin request.
  server: { proxy: { "/api": { target: metricsApi, changeOrigin: true } } },
});
