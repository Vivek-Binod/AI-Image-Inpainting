import { defineConfig, devices } from "@playwright/test";

// End-to-end tests of the static build (npm run build:static), where the networks run in
// the browser via onnxruntime-web. Served by a plain static file server, as on a static host.
//   npx playwright test -c playwright.static.config.ts
// STATIC_URL=https://<deployed site>/ runs the same test against a deployed copy instead.
const PORT = 8766;
const remote = process.env.STATIC_URL;
const libs = process.env.BROWSER_LIBS;
const launchEnv = libs
  ? { ...process.env, LD_LIBRARY_PATH: [libs, process.env.LD_LIBRARY_PATH].filter(Boolean).join(":") }
  : undefined;

export default defineConfig({
  testDir: "./e2e",
  testMatch: /static\.spec/,
  timeout: 180_000,
  workers: 1,
  reporter: [["list"]],
  use: {
    baseURL: remote ?? `http://127.0.0.1:${PORT}`,
    acceptDownloads: true,
    launchOptions: launchEnv ? { env: launchEnv as Record<string, string> } : {},
    screenshot: "only-on-failure",
  },
  projects: [
    { name: "static-desktop", use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 900 } } },
    { name: "static-mobile", use: { ...devices["Pixel 7"] } },
  ],
  webServer: remote
    ? undefined
    : {
        command: `../.venv/bin/python -m http.server ${PORT} --bind 127.0.0.1 --directory dist-static`,
        url: `http://127.0.0.1:${PORT}/index.html`,
        reuseExistingServer: false,
        stdout: "pipe",
      },
});
