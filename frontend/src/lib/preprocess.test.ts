import { describe, expect, it } from "vitest";
import fixtures from "./preprocess.fixtures.json";
import { composite, holeCoverage, predictionToOriginal, resizeF32, resizeU8, toModelInput } from "./preprocess";

// Reference outputs of backend/app/imaging.py (scripts/preprocess_fixtures.py).
const bytes = (s: string) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
const f32 = (s: string) => new Float32Array(bytes(s).buffer);

function maxAbsDiff(a: ArrayLike<number>, b: ArrayLike<number>): number {
  expect(a.length).toBe(b.length);
  let m = 0;
  for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i]));
  return m;
}

describe("browser preprocessing matches the Python backend", () => {
  for (const f of fixtures) {
    it(`${f.W}x${f.H} at work size ${f.work}`, () => {
      const rgb = bytes(f.rgb);
      const mask = bytes(f.mask);
      const x = toModelInput(rgb, mask, f.W, f.H, f.work);
      expect([x.h, x.w]).toEqual(f.hw);
      expect(x.hPad % 8 + (x.wPad % 8)).toBe(0);
      // The hole map must match exactly; values to float32 rounding.
      expect(maxAbsDiff(x.mask, f32(f.modelMask))).toBe(0);
      expect(maxAbsDiff(x.image, f32(f.image))).toBeLessThan(1e-6);
      const full = predictionToOriginal(f32(f.pred), x, f.W, f.H);
      expect(maxAbsDiff(full, bytes(f.full))).toBe(0);
      expect(maxAbsDiff(composite(rgb, full, mask), bytes(f.result))).toBe(0);
    });
  }
});

describe("properties", () => {
  it("hole content never reaches the network", () => {
    const f = fixtures[1];
    const rgb = bytes(f.rgb);
    const mask = bytes(f.mask);
    const other = rgb.slice();
    for (let i = 0; i < mask.length; i++) if (mask[i]) for (let c = 0; c < 3; c++) other[i * 3 + c] = 255 - rgb[i * 3 + c];
    const a = toModelInput(rgb, mask, f.W, f.H, f.work);
    const b = toModelInput(other, mask, f.W, f.H, f.work);
    expect(maxAbsDiff(a.image, b.image)).toBe(0);
  });

  it("known pixels are copied exactly and an empty mask returns the original", () => {
    const f = fixtures[0];
    const rgb = bytes(f.rgb);
    const pred = new Uint8Array(rgb.length).fill(7);
    expect(composite(rgb, pred, new Uint8Array(f.W * f.H))).toEqual(rgb);
  });

  it("resizing to the same size is the identity", () => {
    const src = Float32Array.from({ length: 12 }, (_, i) => i / 7);
    expect(resizeF32(src, 4, 3, 4, 3)).toEqual(src);
    const u8 = Uint8Array.from({ length: 36 }, (_, i) => i * 7);
    expect(resizeU8(u8, 4, 3, 4, 3, 3)).toEqual(u8);
  });

  it("a constant image stays constant after resizing", () => {
    const u8 = new Uint8Array(30 * 20 * 3).fill(200);
    expect(resizeU8(u8, 30, 20, 71, 9, 3).every((v) => v === 200)).toBe(true);
  });

  it("hole coverage marks a single-pixel hole", () => {
    const m = new Uint8Array(16 * 16);
    m[5 * 16 + 9] = 1;
    const cov = holeCoverage(m, 16, 16, 8, 8, 0.5);
    expect(Array.from(cov).reduce((s, v) => s + v, 0)).toBe(1);
    expect(cov[2 * 8 + 4]).toBe(1);
  });
});
