// The mask is kept as a list of operations (so undo/redo is exact) and rasterised
// onto a "committed" canvas at mask resolution. After every operation the touched
// area is binarised (alpha >= 128 -> 255, else 0), so the overlay the user sees is
// exactly the binary mask that is sent: 1 = missing (white), 0 = known (black).

import type { Point, Size } from "./geometry";

export type MaskMode = "paint" | "erase";

export type MaskOp =
  | { kind: "stroke"; mode: MaskMode; size: number; points: Point[] }
  | { kind: "rect"; mode: MaskMode; a: Point; b: Point }
  | { kind: "clear" };

export const MASK_RGB: [number, number, number] = [255, 45, 85];
const MASK_CSS = `rgb(${MASK_RGB.join(",")})`;

interface Bounds {
  x: number;
  y: number;
  w: number;
  h: number;
}

export function opBounds(op: MaskOp, size: Size): Bounds {
  if (op.kind === "clear") return { x: 0, y: 0, w: size.width, h: size.height };
  let x0: number, y0: number, x1: number, y1: number;
  let pad = 2;
  if (op.kind === "rect") {
    x0 = Math.min(op.a.x, op.b.x);
    x1 = Math.max(op.a.x, op.b.x);
    y0 = Math.min(op.a.y, op.b.y);
    y1 = Math.max(op.a.y, op.b.y);
  } else {
    x0 = Math.min(...op.points.map((p) => p.x));
    x1 = Math.max(...op.points.map((p) => p.x));
    y0 = Math.min(...op.points.map((p) => p.y));
    y1 = Math.max(...op.points.map((p) => p.y));
    pad += op.size / 2;
  }
  const bx = Math.max(0, Math.floor(x0 - pad));
  const by = Math.max(0, Math.floor(y0 - pad));
  const ex = Math.min(size.width, Math.ceil(x1 + pad));
  const ey = Math.min(size.height, Math.ceil(y1 + pad));
  return { x: bx, y: by, w: Math.max(0, ex - bx), h: Math.max(0, ey - by) };
}

/** Rectangle snapped to whole pixels so rectangular holes have crisp, exact edges. */
export function rectPixels(a: Point, b: Point, size: Size): Bounds {
  const x0 = clamp(Math.round(Math.min(a.x, b.x)), 0, size.width);
  const x1 = clamp(Math.round(Math.max(a.x, b.x)), 0, size.width);
  const y0 = clamp(Math.round(Math.min(a.y, b.y)), 0, size.height);
  const y1 = clamp(Math.round(Math.max(a.y, b.y)), 0, size.height);
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}

/** Draw one op (without binarising). Also used for the live preview while drawing. */
export function drawOp(ctx: CanvasRenderingContext2D, op: MaskOp, size: Size): void {
  ctx.save();
  if (op.kind === "clear") {
    ctx.clearRect(0, 0, size.width, size.height);
    ctx.restore();
    return;
  }
  ctx.globalCompositeOperation = op.mode === "erase" ? "destination-out" : "source-over";
  ctx.fillStyle = MASK_CSS;
  ctx.strokeStyle = MASK_CSS;
  if (op.kind === "rect") {
    const r = rectPixels(op.a, op.b, size);
    ctx.fillRect(r.x, r.y, r.w, r.h);
  } else if (op.points.length === 1) {
    const p = op.points[0];
    ctx.beginPath();
    ctx.arc(p.x, p.y, op.size / 2, 0, Math.PI * 2);
    ctx.fill();
  } else {
    ctx.lineWidth = op.size;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(op.points[0].x, op.points[0].y);
    for (const p of op.points.slice(1)) ctx.lineTo(p.x, p.y);
    ctx.stroke();
  }
  ctx.restore();
}

/** Make every pixel in ``b`` either fully masked (opaque mask colour) or empty. */
export function binarize(ctx: CanvasRenderingContext2D, b: Bounds): void {
  if (b.w <= 0 || b.h <= 0) return;
  const img = ctx.getImageData(b.x, b.y, b.w, b.h);
  binarizeData(img.data);
  ctx.putImageData(img, b.x, b.y);
}

export function binarizeData(d: Uint8ClampedArray): void {
  const [r, g, bl] = MASK_RGB;
  for (let i = 0; i < d.length; i += 4) {
    if (d[i + 3] >= 128) {
      d[i] = r;
      d[i + 1] = g;
      d[i + 2] = bl;
      d[i + 3] = 255;
    } else {
      d[i] = d[i + 1] = d[i + 2] = d[i + 3] = 0;
    }
  }
}

export function applyOp(ctx: CanvasRenderingContext2D, op: MaskOp, size: Size): void {
  drawOp(ctx, op, size);
  if (op.kind !== "clear") binarize(ctx, opBounds(op, size));
}

export function renderOps(ctx: CanvasRenderingContext2D, ops: MaskOp[], size: Size): void {
  ctx.clearRect(0, 0, size.width, size.height);
  for (const op of ops) applyOp(ctx, op, size);
}

/** RGBA mask-canvas pixels -> grey-level mask bytes (255 = missing, 0 = known) and coverage. */
export function maskBytes(rgba: Uint8ClampedArray): { grey: Uint8ClampedArray; holePixels: number } {
  const n = rgba.length / 4;
  const grey = new Uint8ClampedArray(n);
  let holePixels = 0;
  for (let i = 0; i < n; i++) {
    if (rgba[i * 4 + 3] >= 128) {
      grey[i] = 255;
      holePixels++;
    }
  }
  return { grey, holePixels };
}

/** Encode the committed mask as an opaque black/white PNG (white = missing). */
export async function exportMaskPng(src: HTMLCanvasElement): Promise<{ blob: Blob; holeFraction: number }> {
  const { width, height } = src;
  const data = src.getContext("2d")!.getImageData(0, 0, width, height).data;
  const { grey, holePixels } = maskBytes(data);
  const out = document.createElement("canvas");
  out.width = width;
  out.height = height;
  const octx = out.getContext("2d")!;
  const img = octx.createImageData(width, height);
  for (let i = 0; i < grey.length; i++) {
    const v = grey[i];
    img.data[i * 4] = v;
    img.data[i * 4 + 1] = v;
    img.data[i * 4 + 2] = v;
    img.data[i * 4 + 3] = 255;
  }
  octx.putImageData(img, 0, 0);
  const blob = await new Promise<Blob>((resolve, reject) =>
    out.toBlob((b) => (b ? resolve(b) : reject(new Error("Could not encode the mask."))), "image/png"),
  );
  return { blob, holeFraction: holePixels / (width * height) };
}

export function countHolePixels(src: HTMLCanvasElement): number {
  const data = src.getContext("2d")!.getImageData(0, 0, src.width, src.height).data;
  let n = 0;
  for (let i = 3; i < data.length; i += 4) if (data[i] >= 128) n++;
  return n;
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}
