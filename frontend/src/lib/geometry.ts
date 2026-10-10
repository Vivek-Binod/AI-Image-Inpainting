// Coordinate frames used by the editor:
//   client  - browser pointer coordinates (CSS px, viewport origin)
//   display - position inside the displayed image box (CSS px)
//   mask    - pixels of the mask canvas (the original image size, or a uniformly
//             scaled-down copy for very large photos; same aspect ratio)
//   image   - pixels of the original (EXIF-oriented) image
//   model   - pixels at the resolution the network runs at (longer side = workSize)

export interface Size {
  width: number;
  height: number;
}

export interface Point {
  x: number;
  y: number;
}

export interface BoxLike {
  left: number;
  top: number;
  width: number;
  height: number;
}

// Browsers limit canvas area (iOS Safari: 16,777,216 px). Larger photos get a
// uniformly downscaled mask canvas; the server upsamples it with nearest-neighbour.
export const MAX_MASK_SIDE = 4096;
export const MAX_MASK_AREA = 16_777_216;

export function maskCanvasSize(image: Size, maxSide = MAX_MASK_SIDE, maxArea = MAX_MASK_AREA): Size {
  const { width: w, height: h } = image;
  const scale = Math.min(1, maxSide / Math.max(w, h), Math.sqrt(maxArea / (w * h)));
  if (scale >= 1) return { width: w, height: h };
  return { width: Math.max(1, Math.round(w * scale)), height: Math.max(1, Math.round(h * scale)) };
}

/** Pointer position -> mask-canvas pixel coordinates (unclamped floats). */
export function clientToMask(clientX: number, clientY: number, box: BoxLike, mask: Size): Point {
  return {
    x: ((clientX - box.left) * mask.width) / box.width,
    y: ((clientY - box.top) * mask.height) / box.height,
  };
}

/** Mask-canvas pixel coordinates -> original image pixel coordinates. */
export function maskToImage(p: Point, mask: Size, image: Size): Point {
  return { x: (p.x * image.width) / mask.width, y: (p.y * image.height) / mask.height };
}

/** Length in displayed CSS px -> length in mask pixels (uniform scale). */
export function displayToMaskLength(length: number, box: BoxLike, mask: Size): number {
  return (length * mask.width) / box.width;
}

/** Size the network runs at; mirrors backend/app/imaging.py:model_size (never upscales). */
export function modelSize(image: Size, workSize: number): Size {
  const scale = Math.min(1, workSize / Math.max(image.width, image.height));
  return {
    width: Math.max(8, pyRound(image.width * scale)),
    height: Math.max(8, pyRound(image.height * scale)),
  };
}

// Python's round() uses banker's rounding; match it so the UI shows the server's size.
function pyRound(v: number): number {
  const f = Math.floor(v);
  const d = v - f;
  if (Math.abs(d - 0.5) < 1e-9) return f % 2 === 0 ? f : f + 1;
  return Math.round(v);
}
