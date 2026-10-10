import { execFileSync } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import type { Page } from "@playwright/test";

export const HERE = path.dirname(fileURLToPath(import.meta.url));
export const ROOT = path.resolve(HERE, "../..");
export const PY = path.join(ROOT, ".venv/bin/python");
export const OUT = path.resolve(HERE, "../e2e-artifacts");

export function makeFixtures(): string {
  const dir = path.join(OUT, "fixtures");
  mkdirSync(dir, { recursive: true });
  execFileSync(PY, [path.join(HERE, "fixtures.py"), dir]);
  return dir;
}

/**
 * Record the mask PNG the page sends to /inpaint. Chromium does not expose multipart
 * bodies containing blobs to Playwright, so fetch is wrapped (test instrumentation only;
 * the request still goes to the real backend unchanged).
 */
export async function recordMasks(page: Page) {
  await page.addInitScript(() => {
    const orig = window.fetch.bind(window);
    (window as unknown as { __masks: string[] }).__masks = [];
    window.fetch = async (input: RequestInfo | URL, init?: RequestInit) => {
      if (String(input).endsWith("/inpaint") && init?.body instanceof FormData) {
        const m = init.body.get("mask") as Blob;
        const bytes = new Uint8Array(await m.arrayBuffer());
        let bin = "";
        for (const b of bytes) bin += String.fromCharCode(b);
        (window as unknown as { __masks: string[] }).__masks.push(btoa(bin));
      }
      return orig(input, init);
    };
  });
}

export async function lastMask(page: Page): Promise<Buffer> {
  const b64 = await page.evaluate(() => {
    const m = (window as unknown as { __masks: string[] }).__masks;
    return m[m.length - 1];
  });
  return Buffer.from(b64, "base64");
}

export interface Expect {
  holes: number[][];
  known: number[][];
}

export function verify(original: string, mask: Buffer, result: string, expect: Expect, tag: string) {
  const maskPath = path.join(OUT, `${tag}-mask.png`);
  const expPath = path.join(OUT, `${tag}-expect.json`);
  writeFileSync(maskPath, mask);
  writeFileSync(expPath, JSON.stringify(expect));
  try {
    return JSON.parse(execFileSync(PY, [path.join(HERE, "verify.py"), original, maskPath, result, expPath]).toString());
  } catch (e) {
    const out = (e as { stdout?: Buffer }).stdout?.toString();
    throw new Error(`verification failed for ${tag}: ${out}`);
  }
}

/** Drag with the mouse between two points given as fractions of the canvas box. */
export async function drag(page: Page, from: [number, number], to: [number, number], steps = 12) {
  const box = (await page.getByTestId("mask-canvas").boundingBox())!;
  const p = (f: [number, number]) => [box.x + f[0] * box.width, box.y + f[1] * box.height] as const;
  await page.mouse.move(...p(from));
  await page.mouse.down();
  await page.mouse.move(...p(to), { steps });
  await page.mouse.up();
}

export async function holePercent(page: Page): Promise<number> {
  return parseFloat((await page.getByTestId("hole-fraction").textContent()) ?? "0");
}
