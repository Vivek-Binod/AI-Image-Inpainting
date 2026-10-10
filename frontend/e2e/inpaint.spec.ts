import path from "node:path";
import { expect, test } from "@playwright/test";
import { OUT, drag, holePercent, lastMask, makeFixtures, recordMasks, verify } from "./helpers";

let fixtures: string;
test.beforeAll(() => {
  fixtures = makeFixtures();
});

test("full journey: upload, draw, preview, inpaint with both models, compare, download", async ({ page }) => {
  await recordMasks(page);
  await page.goto("/");
  await expect(page.getByTestId("server-status")).toContainText("2 models");
  await page.screenshot({ path: path.join(OUT, "01-upload.png") });

  const photo = path.join(fixtures, "photo.png");
  await page.getByTestId("file-input").setInputFiles(photo);
  const canvas = page.getByTestId("mask-canvas");
  await expect(canvas).toBeVisible();
  // The mask canvas has the photo's own resolution, whatever its displayed size.
  expect(await canvas.evaluate((c: HTMLCanvasElement) => [c.width, c.height])).toEqual([1024, 768]);
  const box = (await canvas.boundingBox())!;
  expect(box.width).toBeLessThan(1024); // displayed downscaled -> coordinate mapping matters
  await expect(page.getByTestId("inpaint")).toBeDisabled();

  // Rectangle hole.
  await page.getByTestId("tool-rect").click();
  await drag(page, [0.15, 0.2], [0.45, 0.55]);
  const afterRect = await holePercent(page);
  expect(afterRect).toBeGreaterThan(9);
  expect(afterRect).toBeLessThan(12); // 0.30 x 0.35 = 10.5 %

  // Brush stroke.
  await page.getByTestId("tool-brush").click();
  await drag(page, [0.6, 0.3], [0.85, 0.3]);
  const afterBrush = await holePercent(page);
  expect(afterBrush).toBeGreaterThan(afterRect);

  // Eraser band across the rectangle.
  await page.getByTestId("tool-eraser").click();
  await drag(page, [0.1, 0.3], [0.5, 0.3]);
  const afterErase = await holePercent(page);
  expect(afterErase).toBeLessThan(afterBrush);

  // Undo / redo / clear.
  await page.getByTestId("undo").click();
  expect(await holePercent(page)).toBe(afterBrush);
  await page.getByTestId("redo").click();
  expect(await holePercent(page)).toBe(afterErase);
  await page.getByTestId("clear").click();
  expect(await holePercent(page)).toBe(0);
  await expect(page.getByTestId("inpaint")).toBeDisabled();
  await page.keyboard.press("Control+z");
  expect(await holePercent(page)).toBe(afterErase);

  // Overlay toggle.
  await page.getByTestId("toggle-overlay").click();
  await expect(canvas).toHaveCSS("opacity", "0");
  await page.getByTestId("toggle-overlay").click();
  await expect(canvas).toHaveCSS("opacity", "0.6");

  // Masked-input preview is drawn.
  await expect(page.getByTestId("mask-preview")).toBeVisible();
  await page.screenshot({ path: path.join(OUT, "02-editor.png") });

  const expectation = {
    holes: [
      [0.17, 0.22, 0.43, 0.26], // rectangle above the erased band
      [0.17, 0.36, 0.43, 0.53], // rectangle below the erased band
      [0.62, 0.29, 0.83, 0.31], // brush stroke
    ],
    known: [
      [0.2, 0.29, 0.4, 0.31], // erased band inside the rectangle
      [0.55, 0.6, 0.95, 0.95],
      [0.0, 0.0, 0.1, 0.1],
      [0.47, 0.2, 0.57, 0.55], // gap between rectangle and stroke
    ],
  };

  for (const model of ["unet", "gated_unet"]) {
    if (model === "gated_unet") await page.getByRole("button", { name: "Edit mask" }).first().click();
    await page.getByTestId(`model-${model}`).check();
    const reqP = page.waitForRequest((r) => r.url().endsWith("/inpaint") && r.method() === "POST");
    const resP = page.waitForResponse((r) => r.url().endsWith("/inpaint"));
    await page.getByTestId("inpaint").click();
    await reqP;
    const res = await resP;
    expect(res.status()).toBe(200);
    expect(res.headers()["x-model"]).toBe(model);
    expect(res.headers()["x-image-size"]).toBe("1024x768");
    expect(res.headers()["x-work-size"]).toBe("128x96");
    await expect(page.getByTestId("result-view")).toBeVisible();
    await expect(page.getByTestId("compare")).toBeVisible();
    await expect(page.getByTestId("inpaint")).toBeEnabled();
    await page.screenshot({ path: path.join(OUT, `03-result-${model}.png`) });

    const dl = page.waitForEvent("download");
    await page.getByTestId("download").click();
    const download = await dl;
    expect(download.suggestedFilename()).toBe(`photo_inpainted_${model}.png`);
    const saved = path.join(OUT, `result-${model}.png`);
    await download.saveAs(saved);

    const report = verify(photo, await lastMask(page), saved, expectation, `desktop-${model}`);
    expect(report.ok, JSON.stringify(report)).toBe(true);
    expect(report.mask_size).toEqual([1024, 768]);
  }

  // Side-by-side view shows the full-resolution result.
  await page.getByRole("tab", { name: "Side by side" }).click();
  const natural = await page.getByTestId("result-image").evaluate((i: HTMLImageElement) => [i.naturalWidth, i.naturalHeight]);
  expect(natural).toEqual([1024, 768]);
  await page.screenshot({ path: path.join(OUT, "04-side-by-side.png") });

  // Reset returns to the upload screen.
  await page.getByTestId("reset").click();
  await expect(page.getByTestId("dropzone")).toBeVisible();
});

test("EXIF-rotated JPEG: the mask drawn on the displayed image lines up on the server", async ({ page }) => {
  await recordMasks(page);
  await page.goto("/");
  const jpg = path.join(fixtures, "rotated.jpg");
  await page.getByTestId("file-input").setInputFiles(jpg);
  const canvas = page.getByTestId("mask-canvas");
  // Stored 768x600 with orientation 6 -> displayed portrait 600x768.
  expect(await canvas.evaluate((c: HTMLCanvasElement) => [c.width, c.height])).toEqual([600, 768]);
  await page.getByTestId("tool-rect").click();
  await drag(page, [0.2, 0.1], [0.8, 0.3]);
  const reqP = page.waitForRequest((r) => r.url().endsWith("/inpaint"));
  await page.getByTestId("inpaint").click();
  await reqP;
  await expect(page.getByTestId("result-view")).toBeVisible();
  const dl = page.waitForEvent("download");
  await page.getByTestId("download").click();
  const saved = path.join(OUT, "result-rotated.png");
  await (await dl).saveAs(saved);
  const report = verify(jpg, await lastMask(page), saved, {
    holes: [[0.22, 0.12, 0.78, 0.28]],
    known: [[0, 0.4, 1, 1], [0, 0, 0.15, 1]],
  }, "rotated");
  expect(report.ok, JSON.stringify(report)).toBe(true);
  expect(report.size).toEqual([600, 768]);
});

test("unreadable upload shows an error", async ({ page }) => {
  await recordMasks(page);
  await page.goto("/");
  await page.getByTestId("file-input").setInputFiles({ name: "broken.png", mimeType: "image/png", buffer: Buffer.from("not a png") });
  await expect(page.getByTestId("upload-error")).toContainText("cannot display");
});

test("server error is shown and no result is displayed (simulated failure)", async ({ page }) => {
  await recordMasks(page);
  await page.goto("/");
  await page.getByTestId("file-input").setInputFiles(path.join(fixtures, "photo.png"));
  await page.getByTestId("tool-rect").click();
  await drag(page, [0.3, 0.3], [0.5, 0.5]);
  // The only mocked response in the suite: simulate a backend failure.
  await page.route("**/inpaint", (route) =>
    route.fulfill({ status: 503, contentType: "application/json", body: JSON.stringify({ detail: "The U-Net checkpoint is unavailable: simulated" }) }),
  );
  await page.getByTestId("inpaint").click();
  await expect(page.getByTestId("error")).toContainText("checkpoint is unavailable");
  await expect(page.getByRole("tab", { name: "Result" })).toBeDisabled();
});
