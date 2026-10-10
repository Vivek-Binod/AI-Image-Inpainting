// Static build: the network runs in the visitor's browser (see inpaint.worker.ts), so no
// image ever leaves the device. Exposes the same interface as the server API in api.ts.
import type { InpaintResponse, ModelInfo } from "./api";
import { ApiError } from "./api";
import type { WorkerMessage, WorkerRequest } from "./inpaint.worker";

export interface Progress {
  stage: "download" | "run";
  loaded?: number;
  total?: number;
}

const asset = (path: string) => new URL(path, document.baseURI).href;

export async function getModelInfo(signal?: AbortSignal): Promise<ModelInfo> {
  const res = await fetch(asset("models/models.json"), { signal }).catch(() => null);
  if (!res?.ok) throw new ApiError("Could not load the model list.");
  return res.json();
}

let worker: Worker | null = null;
let nextId = 0;

function getWorker(): Worker {
  worker ??= new Worker(new URL("./inpaint.worker.ts", import.meta.url), { type: "module" });
  return worker;
}

/** Pixels of the image as displayed (EXIF orientation applied, alpha flattened on white),
 * drawn at ``size`` - the mask resolution. */
function imagePixels(el: HTMLImageElement, size: { width: number; height: number }): Uint8Array {
  const c = document.createElement("canvas");
  c.width = size.width;
  c.height = size.height;
  const ctx = c.getContext("2d", { willReadFrequently: true })!;
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, size.width, size.height);
  ctx.drawImage(el, 0, 0, size.width, size.height);
  const rgba = ctx.getImageData(0, 0, size.width, size.height).data;
  const rgb = new Uint8Array(size.width * size.height * 3);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
    rgb[j] = rgba[i];
    rgb[j + 1] = rgba[i + 1];
    rgb[j + 2] = rgba[i + 2];
  }
  return rgb;
}

function maskPixels(maskCanvas: HTMLCanvasElement): Uint8Array {
  const { width, height } = maskCanvas;
  const rgba = maskCanvas.getContext("2d")!.getImageData(0, 0, width, height).data;
  const m = new Uint8Array(width * height);
  for (let i = 0; i < m.length; i++) m[i] = rgba[i * 4 + 3] >= 128 ? 1 : 0; // same rule as exportMaskPng
  return m;
}

async function encodePng(rgb: Uint8Array, width: number, height: number): Promise<Blob> {
  const c = document.createElement("canvas");
  c.width = width;
  c.height = height;
  const ctx = c.getContext("2d")!;
  const img = ctx.createImageData(width, height);
  for (let i = 0, j = 0; j < rgb.length; i += 4, j += 3) {
    img.data[i] = rgb[j];
    img.data[i + 1] = rgb[j + 1];
    img.data[i + 2] = rgb[j + 2];
    img.data[i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return new Promise((resolve, reject) =>
    c.toBlob((b) => (b ? resolve(b) : reject(new ApiError("Could not encode the result."))), "image/png"),
  );
}

export async function inpaintInBrowser(
  el: HTMLImageElement,
  maskCanvas: HTMLCanvasElement,
  model: string,
  workSize: number,
  signal: AbortSignal,
  onProgress: (p: Progress) => void,
): Promise<InpaintResponse> {
  const { width, height } = maskCanvas;
  const t0 = performance.now();
  const rgb = imagePixels(el, { width, height });
  const mask = maskPixels(maskCanvas);
  let holes = 0;
  for (let i = 0; i < mask.length; i++) holes += mask[i];
  const prepMs = performance.now() - t0;
  let runStart = performance.now();
  const id = ++nextId;
  const w = getWorker();
  const msg = await new Promise<Extract<WorkerMessage, { kind: "done" }>>((resolve, reject) => {
    const onMessage = (e: MessageEvent<WorkerMessage>) => {
      const m = e.data;
      if (m.id !== id) return;
      if (m.kind === "progress") {
        if (m.stage === "run") runStart = performance.now(); // time excludes the one-off model download
        onProgress(m);
      }
      else {
        cleanup();
        if (m.kind === "done") resolve(m);
        else reject(new ApiError(`In-browser inference failed: ${m.message}`));
      }
    };
    const onError = (e: ErrorEvent) => {
      cleanup();
      worker?.terminate();
      worker = null;
      reject(new ApiError(`In-browser inference failed: ${e.message || "the worker crashed"}`));
    };
    // The run itself cannot be interrupted; a cancelled request's result is ignored.
    const onAbort = () => {
      cleanup();
      reject(new DOMException("Cancelled", "AbortError"));
    };
    const cleanup = () => {
      w.removeEventListener("message", onMessage);
      w.removeEventListener("error", onError);
      signal.removeEventListener("abort", onAbort);
    };
    w.addEventListener("message", onMessage);
    w.addEventListener("error", onError);
    signal.addEventListener("abort", onAbort);
    const req: WorkerRequest = { id, modelUrl: asset(`models/${model}.onnx`), rgb, mask, width, height, workSize };
    w.postMessage(req, [rgb.buffer, mask.buffer]);
  });
  return {
    blob: await encodePng(msg.result, width, height),
    model,
    workSize: msg.workSize,
    imageSize: `${width}x${height}`,
    holeFraction: holes / (width * height),
    inferenceMs: msg.inferenceMs,
    totalMs: prepMs + performance.now() - runStart,
  };
}
