// In-browser port of backend/app/imaging.py (to_model_input, prediction_to_original,
// composite) for the static build, where the network runs in the browser. Resizing
// reimplements Pillow's bicubic resampling (separable, horizontal then vertical, a = -0.5,
// support scaled when downscaling) so the network sees the same input as on the server;
// src/lib/preprocess.test.ts checks this against fixtures produced by the Python code.
import { modelSize } from "./geometry";

const PRECISION_BITS = 22; // Pillow: 32 - 8 - 2
const BICUBIC_SUPPORT = 2;

function bicubic(x: number): number {
  const a = -0.5;
  if (x < 0) x = -x;
  if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
  if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
  return 0;
}

interface Coeffs {
  ksize: number;
  bounds: Int32Array; // [xmin, count] per output pixel
  k: Float64Array; // ksize weights per output pixel
}

function coeffs(inSize: number, outSize: number): Coeffs {
  const scale = inSize / outSize;
  const filterscale = Math.max(scale, 1);
  const support = BICUBIC_SUPPORT * filterscale;
  const ksize = Math.ceil(support) * 2 + 1;
  const bounds = new Int32Array(outSize * 2);
  const k = new Float64Array(outSize * ksize);
  const ss = 1 / filterscale;
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    const xmin = Math.max(Math.trunc(center - support + 0.5), 0);
    const xmax = Math.min(Math.trunc(center + support + 0.5), inSize) - xmin;
    let ww = 0;
    for (let x = 0; x < xmax; x++) {
      const w = bicubic((x + xmin - center + 0.5) * ss);
      k[xx * ksize + x] = w;
      ww += w;
    }
    if (ww !== 0) for (let x = 0; x < xmax; x++) k[xx * ksize + x] /= ww;
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  return { ksize, bounds, k };
}

/** Pillow ``Image.resize(BICUBIC)`` of a single-channel float32 ("F") image. */
export function resizeF32(src: Float32Array, W: number, H: number, w: number, h: number): Float32Array {
  let cur = src;
  let cw = W;
  if (w !== W) {
    const c = coeffs(W, w);
    const out = new Float32Array(w * H);
    for (let y = 0; y < H; y++) {
      for (let xx = 0; xx < w; xx++) {
        const xmin = c.bounds[xx * 2], n = c.bounds[xx * 2 + 1], o = xx * c.ksize;
        let ss = 0;
        for (let x = 0; x < n; x++) ss += cur[y * cw + x + xmin] * c.k[o + x];
        out[y * w + xx] = ss;
      }
    }
    cur = out;
    cw = w;
  }
  if (h !== H) {
    const c = coeffs(H, h);
    const out = new Float32Array(cw * h);
    for (let yy = 0; yy < h; yy++) {
      const ymin = c.bounds[yy * 2], n = c.bounds[yy * 2 + 1], o = yy * c.ksize;
      for (let x = 0; x < cw; x++) {
        let ss = 0;
        for (let y = 0; y < n; y++) ss += cur[(y + ymin) * cw + x] * c.k[o + y];
        out[yy * cw + x] = ss;
      }
    }
    cur = out;
  }
  return cur === src ? src.slice() : cur;
}

function fixedPoint(c: Coeffs): Int32Array {
  const kk = new Int32Array(c.k.length);
  const one = 1 << PRECISION_BITS;
  for (let i = 0; i < c.k.length; i++) {
    const v = c.k[i];
    kk[i] = v < 0 ? Math.trunc(-0.5 + v * one) : Math.trunc(0.5 + v * one);
  }
  return kk;
}

function clip8(ss: number): number {
  if (ss >= 2 ** (PRECISION_BITS + 8)) return 255;
  if (ss <= 0) return 0;
  return Math.floor(ss / 2 ** PRECISION_BITS);
}

/** Pillow ``Image.resize(BICUBIC)`` of an 8-bit image with ``ch`` interleaved channels. */
export function resizeU8(src: Uint8Array, W: number, H: number, w: number, h: number, ch: number): Uint8Array {
  const half = 1 << (PRECISION_BITS - 1);
  let cur = src;
  let cw = W;
  if (w !== W) {
    const c = coeffs(W, w);
    const kk = fixedPoint(c);
    const out = new Uint8Array(w * H * ch);
    for (let y = 0; y < H; y++) {
      for (let xx = 0; xx < w; xx++) {
        const xmin = c.bounds[xx * 2], n = c.bounds[xx * 2 + 1], o = xx * c.ksize;
        for (let b = 0; b < ch; b++) {
          let ss = half;
          for (let x = 0; x < n; x++) ss += cur[(y * cw + x + xmin) * ch + b] * kk[o + x];
          out[(y * w + xx) * ch + b] = clip8(ss);
        }
      }
    }
    cur = out;
    cw = w;
  }
  if (h !== H) {
    const c = coeffs(H, h);
    const kk = fixedPoint(c);
    const out = new Uint8Array(cw * h * ch);
    for (let yy = 0; yy < h; yy++) {
      const ymin = c.bounds[yy * 2], n = c.bounds[yy * 2 + 1], o = yy * c.ksize;
      for (let x = 0; x < cw; x++) {
        for (let b = 0; b < ch; b++) {
          let ss = half;
          for (let y = 0; y < n; y++) ss += cur[((y + ymin) * cw + x) * ch + b] * kk[o + y];
          out[(yy * cw + x) * ch + b] = clip8(ss);
        }
      }
    }
    cur = out;
  }
  return cur === src ? src.slice() : cur;
}

/** Low-resolution hole map: true where any full-resolution hole pixel lies within ``radius``
 * low-resolution pixels of the centre (port of ``_hole_coverage``, via an integral image). */
export function holeCoverage(mask: Uint8Array, W: number, H: number, w: number, h: number, radius: number): Uint8Array {
  const integral = new Float64Array((H + 1) * (W + 1));
  for (let y = 0; y < H; y++) {
    let row = 0;
    for (let x = 0; x < W; x++) {
      row += mask[y * W + x] ? 1 : 0;
      integral[(y + 1) * (W + 1) + x + 1] = integral[y * (W + 1) + x + 1] + row;
    }
  }
  const ranges = (nOut: number, nIn: number) => {
    const scale = nIn / nOut;
    const lo = new Int32Array(nOut), hi = new Int32Array(nOut);
    for (let i = 0; i < nOut; i++) {
      const c = (i + 0.5) * scale;
      lo[i] = Math.min(Math.max(Math.floor(c - radius * scale), 0), nIn);
      hi[i] = Math.min(Math.max(Math.ceil(c + radius * scale), 0), nIn);
    }
    return [lo, hi];
  };
  const [y0, y1] = ranges(h, H);
  const [x0, x1] = ranges(w, W);
  const out = new Uint8Array(w * h);
  const I = (y: number, x: number) => integral[y * (W + 1) + x];
  for (let i = 0; i < h; i++)
    for (let j = 0; j < w; j++)
      out[i * w + j] = I(y1[i], x1[j]) - I(y0[i], x1[j]) - I(y1[i], x0[j]) + I(y0[i], x0[j]) > 0 ? 1 : 0;
  return out;
}

export interface ModelInput {
  /** (3, hPad, wPad) planar float32 in [0, 1], holes zeroed. */
  image: Float32Array;
  /** (hPad, wPad) float32 in {0, 1}. */
  mask: Float32Array;
  h: number;
  w: number;
  hPad: number;
  wPad: number;
}

function reflectIndex(i: number, n: number, mode: "reflect" | "edge"): number {
  if (i < n) return i;
  return mode === "reflect" ? 2 * (n - 1) - i : n - 1;
}

/** Port of ``to_model_input(..., resample="normalized")``. ``rgb`` is (H, W, 3) uint8 and
 * ``mask`` is (H, W) with 1 = missing. */
export function toModelInput(rgb: Uint8Array, mask: Uint8Array, W: number, H: number, workSize: number): ModelInput {
  const { width: w, height: h } = modelSize({ width: W, height: H }, workSize);
  const img = [new Float32Array(w * h), new Float32Array(w * h), new Float32Array(w * h)];
  const m = new Float32Array(w * h);
  if (w === W && h === H) {
    for (let i = 0; i < w * h; i++) {
      for (let c = 0; c < 3; c++) img[c][i] = rgb[i * 3 + c] / 255;
      m[i] = mask[i] ? 1 : 0;
    }
  } else {
    const known = new Float32Array(W * H);
    for (let i = 0; i < W * H; i++) known[i] = mask[i] ? 0 : 1;
    const den = resizeF32(known, W, H, w, h);
    const chan = new Float32Array(W * H);
    for (let c = 0; c < 3; c++) {
      for (let i = 0; i < W * H; i++) chan[i] = Math.fround(rgb[i * 3 + c] / 255) * known[i];
      const r = resizeF32(chan, W, H, w, h);
      for (let i = 0; i < w * h; i++) img[c][i] = Math.min(Math.max(Math.fround(r[i] / Math.max(den[i], Math.fround(1e-6))), 0), 1);
    }
    const cov = holeCoverage(mask, W, H, w, h, 0.5);
    for (let i = 0; i < w * h; i++) m[i] = cov[i] || den[i] < 0.5 ? 1 : 0;
  }
  for (let c = 0; c < 3; c++) for (let i = 0; i < w * h; i++) img[c][i] = img[c][i] * (1 - m[i]);

  const hPad = h + ((8 - (h % 8)) % 8);
  const wPad = w + ((8 - (w % 8)) % 8);
  const mode = h > hPad - h && w > wPad - w ? "reflect" : "edge";
  const image = new Float32Array(3 * hPad * wPad);
  const maskPad = new Float32Array(hPad * wPad);
  for (let y = 0; y < hPad; y++) {
    const sy = reflectIndex(y, h, mode);
    for (let x = 0; x < wPad; x++) {
      const s = sy * w + reflectIndex(x, w, mode);
      const d = y * wPad + x;
      maskPad[d] = m[s];
      for (let c = 0; c < 3; c++) image[c * hPad * wPad + d] = img[c][s];
    }
  }
  return { image, mask: maskPad, h, w, hPad, wPad };
}

/** numpy ``round`` (half to even). */
function roundHalfEven(v: number): number {
  const r = Math.round(v);
  return Math.abs(v % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** Port of ``prediction_to_original``: crop the padding, quantise and bicubic-resize the
 * (3, hPad, wPad) prediction to (H, W, 3) uint8. */
export function predictionToOriginal(pred: Float32Array, x: ModelInput, W: number, H: number): Uint8Array {
  const { h, w, hPad, wPad } = x;
  const small = new Uint8Array(w * h * 3);
  for (let y = 0; y < h; y++)
    for (let xx = 0; xx < w; xx++)
      for (let c = 0; c < 3; c++) {
        const v = Math.min(Math.max(pred[c * hPad * wPad + y * wPad + xx], 0), 1);
        small[(y * w + xx) * 3 + c] = roundHalfEven(Math.fround(v * 255));
      }
  return w === W && h === H ? small : resizeU8(small, w, h, W, H, 3);
}

/** result = I * (1 - M) + P * M on (H, W, 3) uint8: known pixels stay bit-identical. */
export function composite(original: Uint8Array, prediction: Uint8Array, mask: Uint8Array): Uint8Array {
  const out = original.slice();
  for (let i = 0; i < mask.length; i++)
    if (mask[i]) for (let c = 0; c < 3; c++) out[i * 3 + c] = prediction[i * 3 + c];
  return out;
}
