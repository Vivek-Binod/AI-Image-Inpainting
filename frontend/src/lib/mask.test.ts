import { describe, expect, it } from "vitest";
import { binarizeData, maskBytes, opBounds, rectPixels } from "./mask";

describe("binary mask encoding", () => {
  it("binarises alpha at 128 and never produces grey levels", () => {
    const d = new Uint8ClampedArray([9, 9, 9, 127, 9, 9, 9, 128, 0, 0, 0, 0, 1, 2, 3, 255]);
    binarizeData(d);
    expect(Array.from(d.filter((_, i) => i % 4 === 3))).toEqual([0, 255, 0, 255]);
  });

  it("exports painted pixels as white (missing = 1) and the rest as black (known = 0)", () => {
    const rgba = new Uint8ClampedArray([255, 45, 85, 255, 0, 0, 0, 0, 255, 45, 85, 255]);
    const { grey, holePixels } = maskBytes(rgba);
    expect(Array.from(grey)).toEqual([255, 0, 255]);
    expect(holePixels).toBe(2);
  });
});

describe("rectangles", () => {
  const size = { width: 100, height: 80 };
  it("snaps to whole pixels in any drag direction and clamps to the canvas", () => {
    expect(rectPixels({ x: 10.4, y: 20.6 }, { x: 30.5, y: 5.2 }, size)).toEqual({ x: 10, y: 5, w: 21, h: 16 });
    expect(rectPixels({ x: -20, y: -5 }, { x: 150, y: 90 }, size)).toEqual({ x: 0, y: 0, w: 100, h: 80 });
  });
  it("bounds include the brush radius", () => {
    const b = opBounds({ kind: "stroke", mode: "paint", size: 10, points: [{ x: 50, y: 40 }] }, size);
    expect(b.x).toBeLessThanOrEqual(45);
    expect(b.x + b.w).toBeGreaterThanOrEqual(55);
  });
});
