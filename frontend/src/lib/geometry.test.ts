import { describe, expect, it } from "vitest";
import { clientToMask, displayToMaskLength, maskCanvasSize, maskToImage, modelSize } from "./geometry";

describe("pointer -> mask -> image mapping", () => {
  // A 4032x3024 photo shown 806.4 px wide at (100, 50) on the page.
  const image = { width: 4032, height: 3024 };
  const box = { left: 100, top: 50, width: 806.4, height: 604.8 };

  it("maps the corners and centre of the displayed image exactly", () => {
    expect(clientToMask(100, 50, box, image)).toEqual({ x: 0, y: 0 });
    expect(clientToMask(906.4, 654.8, box, image)).toEqual({ x: 4032, y: 3024 });
    const c = clientToMask(100 + 403.2, 50 + 302.4, box, image);
    expect(c.x).toBeCloseTo(2016, 6);
    expect(c.y).toBeCloseTo(1512, 6);
  });

  it("keeps x and y independent (no transposition) for non-square images", () => {
    const p = clientToMask(100 + 80.64, 50, box, image);
    expect(p.x).toBeCloseTo(403.2, 6);
    expect(p.y).toBe(0);
  });

  it("maps through a downscaled mask canvas back to original pixels", () => {
    const big = { width: 8000, height: 6000 };
    const mask = maskCanvasSize(big);
    expect(mask.width / mask.height).toBeCloseTo(8000 / 6000, 3);
    const m = clientToMask(100 + 403.2, 50 + 302.4, box, mask);
    const img = maskToImage(m, mask, big);
    expect(img.x).toBeCloseTo(4000, 6);
    expect(img.y).toBeCloseTo(3000, 6);
  });

  it("converts brush diameter from screen pixels to mask pixels", () => {
    expect(displayToMaskLength(40, box, image)).toBeCloseTo(200, 6);
  });
});

describe("mask canvas size", () => {
  it("is the image size for ordinary photos", () => {
    expect(maskCanvasSize({ width: 4032, height: 3024 })).toEqual({ width: 4032, height: 3024 });
    expect(maskCanvasSize({ width: 256, height: 256 })).toEqual({ width: 256, height: 256 });
  });
  it("is capped by side and area, keeping the aspect ratio", () => {
    const s = maskCanvasSize({ width: 9000, height: 3000 });
    expect(s.width).toBe(4096);
    expect(s.height).toBe(1365);
    const a = maskCanvasSize({ width: 6000, height: 6000 }, 10000, 16_777_216);
    expect(a.width * a.height).toBeLessThanOrEqual(16_777_216);
  });
});

describe("model size mirrors the backend", () => {
  it.each([
    [256, 200, 128, 128, 100],
    [256, 256, 128, 128, 128],
    [4032, 3024, 128, 128, 96],
    [60, 100, 128, 60, 100],
    [1000, 4000, 256, 64, 256],
    [301, 517, 128, 75, 128], // 74.52 -> 75
  ])("%ix%i @ %i -> %ix%i", (w, h, work, ew, eh) => {
    expect(modelSize({ width: w, height: h }, work)).toEqual({ width: ew, height: eh });
  });
});
