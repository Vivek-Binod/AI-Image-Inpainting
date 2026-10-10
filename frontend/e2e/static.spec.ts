import { execFileSync } from "node:child_process";
import { writeFileSync } from "node:fs";
import path from "node:path";
import { expect, test } from "@playwright/test";
import type { Page } from "@playwright/test";
import { HERE, OUT, PY, drag, makeFixtures, verify } from "./helpers";

// Static build: no backend. The mask is read back from the drawing canvas (same rule as the
// exported mask: alpha >= 128), and the in-browser result is checked against verify.py and
// against the Python server code run on the same image and mask.

async function canvasMask(page: Page): Promise<Buffer> {
  const url = await page.getByTestId("mask-canvas").evaluate((c: HTMLCanvasElement) => {
    const d = c.getContext("2d")!.getImageData(0, 0, c.width, c.height);
    const out = document.createElement("canvas");
    out.width = c.width;
    out.height = c.height;
    const ctx = out.getContext("2d")!;
    const img = ctx.createImageData(c.width, c.height);
    for (let i = 0; i < d.data.length; i += 4) {
      const v = d.data[i + 3] >= 128 ? 255 : 0;
      img.data[i] = img.data[i + 1] = img.data[i + 2] = v;
      img.data[i + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    return out.toDataURL("image/png");
  });
  return Buffer.from(url.split(",")[1], "base64");
}

function compareWithServer(photo: string, mask: Buffer, result: string, model: string, tag: string) {
  const maskPath = path.join(OUT, `${tag}-mask.png`);
  writeFileSync(maskPath, mask);
  return JSON.parse(execFileSync(PY, [path.join(HERE, "compare_server.py"), photo, maskPath, result, model, "128"]).toString());
}

test("static build: models run in the browser and match the server", async ({ page, isMobile }) => {
  const fixtures = makeFixtures();
  const photo = path.join(fixtures, "photo.png");
  const requests: string[] = [];
  const uploads: string[] = []; // any request carrying a body (the photo or mask would need one)
  page.on("request", (r) => {
    requests.push(new URL(r.url()).pathname);
    if (r.method() !== "GET" || r.postDataBuffer()) uploads.push(`${r.method()} ${r.url()}`);
  });
  page.on("pageerror", (e) => console.log("pageerror:", e.message));

  await page.goto("/");
  await expect(page.getByTestId("server-status")).toContainText("2 models · in browser");
  await page.getByTestId("file-input").setInputFiles(photo);
  const canvas = page.getByTestId("mask-canvas");
  await expect(canvas).toBeVisible();
  await canvas.scrollIntoViewIfNeeded();
  await page.getByTestId("tool-rect").click();
  await drag(page, [0.15, 0.2], [0.45, 0.55]);
  await page.getByTestId("tool-brush").click();
  await drag(page, [0.6, 0.3], [0.85, 0.3]);
  const mask = await canvasMask(page);
  const expectation = {
    holes: [[0.17, 0.22, 0.43, 0.53], [0.62, 0.29, 0.83, 0.31]],
    known: [[0.55, 0.6, 0.95, 0.95], [0.0, 0.0, 0.1, 0.1], [0.47, 0.42, 0.57, 0.55]],
  };
  const models = isMobile ? ["gated_unet"] : ["unet", "gated_unet"];
  for (const model of models) {
    if (model === "gated_unet" && !isMobile) await page.getByRole("button", { name: "Edit mask" }).first().click();
    await page.getByTestId(`model-${model}`).check();
    await page.getByTestId("inpaint").click();
    await expect(page.getByTestId("result-view")).toBeVisible({ timeout: 120_000 });
    await expect(page.getByRole("status")).toContainText("at 128x96");
    await page.screenshot({ path: path.join(OUT, `static-${isMobile ? "mobile" : "desktop"}-${model}.png`), fullPage: isMobile });
    const dl = page.waitForEvent("download");
    await page.getByTestId("download").click();
    const saved = path.join(OUT, `static-result-${model}${isMobile ? "-mobile" : ""}.png`);
    await (await dl).saveAs(saved);

    const tag = `static-${isMobile ? "mobile" : "desktop"}-${model}`;
    const report = verify(photo, mask, saved, expectation, tag);
    expect(report.ok, JSON.stringify(report)).toBe(true);
    const cmp = compareWithServer(photo, mask, saved, model, tag);
    console.log(tag, JSON.stringify(cmp));
    expect(cmp.known_identical).toBe(true);
    expect(cmp.max_hole_diff).toBeLessThanOrEqual(3);
  }
  // Only GET requests without a body: the image and mask never left the browser.
  expect(uploads).toEqual([]);
  expect(requests).toContain("/models/gated_unet.onnx");
  expect(await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)).toBeLessThanOrEqual(0);
});
