import { defineConfig, devices } from "@playwright/test";

// End-to-end tests drive the real website against the real FastAPI backend and the
// trained checkpoints (nothing is mocked except the one simulated-failure test).
// Build the frontend first (npm run build); the backend serves it on :8765.
//   npx playwright test
// BROWSER_LIBS may point at extra shared libraries for Chromium (e.g. a local libgbm).
const PORT = 8765;
const libs = process.env.BROWSER_LIBS;
const launchEnv = libs
  ? { ...process.env, LD_LIBRARY_PATH: [libs, process.env.LD_LIBRARY_PATH].filter(Boolean).join(":") }
  : undefined;

export default defineConfig({
  testDir: "./e2e",
  timeout: 120_000,
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    acceptDownloads: true,
    launchOptions: launchEnv ? { env: launchEnv as Record<string, string> } : {},
    screenshot: "only-on-failure",
  },
  projects: [
    { name: "desktop", use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } }, testIgnore: /mobile|static/ },
    { name: "mobile", use: { ...devices["Pixel 7"] }, testMatch: /mobile/ },
  ],
  webServer: {
    command: `.venv/bin/python -m uvicorn backend.app.main:app --host 127.0.0.1 --port ${PORT}`,
    cwd: "..",
    url: `http://127.0.0.1:${PORT}/health`,
    timeout: 120_000,
    reuseExistingServer: false,
    env: { INPAINT_FRONTEND_DIST: "frontend/dist" },
    stdout: "pipe",
  },
});
