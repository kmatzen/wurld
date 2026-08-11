// Browser tests for the viewer. See test/viewer/viewer.spec.mjs for why these
// need a real browser rather than jsdom.
import { defineConfig, devices } from '@playwright/test';

const PORT = Number(process.env.VIEWER_TEST_PORT || 8791);

export default defineConfig({
  testDir: './test/viewer',
  testMatch: '**/*.spec.mjs',
  // The viewer decodes real video; a slow CI runner should not read as a bug.
  timeout: 60_000,
  expect: { timeout: 15_000 },
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: 0,
  reporter: process.env.CI ? 'list' : 'line',
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    trace: 'retain-on-failure',
  },
  projects: [
    // Chromium only: decoding goes through WebCodecs, which Firefox and WebKit
    // do not implement compatibly enough for the chromapakz decoder path.
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
  ],
  webServer: {
    command: 'node test/viewer/serve.mjs',
    url: `http://127.0.0.1:${PORT}/viewer/index.html`,
    reuseExistingServer: !process.env.CI,
    stdout: 'ignore',
    stderr: 'pipe',
  },
});
