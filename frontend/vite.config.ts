import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// In development the API runs separately on :8000; these paths are proxied to it so
// the frontend uses the same relative URLs in development and production.
const api = process.env.VITE_API_PROXY ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  // The in-browser engine's worker imports onnxruntime-web, which needs an ES-module worker.
  worker: { format: "es" },
  server: {
    host: true,
    port: 5173,
    proxy: {
      "/health": api,
      "/model": api,
      "/inpaint": api,
    },
  },
  test: {
    environment: "jsdom",
    include: ["src/**/*.test.ts"],
  },
});
