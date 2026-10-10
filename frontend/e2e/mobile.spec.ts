import path from "node:path";
import { expect, test } from "@playwright/test";
import { OUT, lastMask, makeFixtures, recordMasks, verify } from "./helpers";

test("mobile: touch drawing, inpaint and result without horizontal scrolling", async ({ page }) => {
  const fixtures = makeFixtures();
  await recordMasks(page);
  await page.goto("/");
  const photo = path.join(fixtures, "photo.png");
  await page.getByTestId("file-input").setInputFiles(photo);
  const canvas = page.getByTestId("mask-canvas");
  await expect(canvas).toBeVisible();
  await canvas.scrollIntoViewIfNeeded();
  const box = (await canvas.boundingBox())!;

  // A real touch stroke (touchstart/move/end -> pointer events with pointerType "touch").
  const cdp = await page.context().newCDPSession(page);
  const at = (fx: number, fy: number) => [{ x: box.x + fx * box.width, y: box.y + fy * box.height }];
  await cdp.send("Input.dispatchTouchEvent", { type: "touchStart", touchPoints: at(0.2, 0.5) });
  for (let i = 1; i <= 15; i++) {
    await cdp.send("Input.dispatchTouchEvent", { type: "touchMove", touchPoints: at(0.2 + (0.5 * i) / 15, 0.5) });
  }
  await cdp.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
  const pct = parseFloat((await page.getByTestId("hole-fraction").textContent()) ?? "0");
  expect(pct).toBeGreaterThan(1);

  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBeLessThanOrEqual(0);
  await page.screenshot({ path: path.join(OUT, "05-mobile-editor.png"), fullPage: true });

  const reqP = page.waitForRequest((r) => r.url().endsWith("/inpaint"));
  await page.getByTestId("inpaint").click();
  await reqP;
  await expect(page.getByTestId("result-view")).toBeVisible();
  await page.screenshot({ path: path.join(OUT, "06-mobile-result.png"), fullPage: true });
  const dl = page.waitForEvent("download");
  await page.getByTestId("download").click();
  const saved = path.join(OUT, "result-mobile.png");
  await (await dl).saveAs(saved);
  const report = verify(photo, await lastMask(page), saved, {
    holes: [[0.25, 0.495, 0.65, 0.505]],
    known: [[0, 0, 1, 0.4], [0, 0.6, 1, 1]],
  }, "mobile");
  expect(report.ok, JSON.stringify(report)).toBe(true);
  expect(await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth)).toBeLessThanOrEqual(0);
});
