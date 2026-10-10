/// <reference lib="webworker" />
// Runs the exported ONNX network in a Web Worker for the static (in-browser) build:
// preprocessing, inference, upscaling and compositing all happen here, off the UI thread.
import * as ort from "onnxruntime-web/wasm";
import { composite, predictionToOriginal, toModelInput } from "./preprocess";

export interface WorkerRequest {
  id: number;
  modelUrl: string;
  rgb: Uint8Array; // (H, W, 3)
  mask: Uint8Array; // (H, W), 1 = missing
  width: number;
  height: number;
  workSize: number;
}

export type WorkerMessage =
  | { id: number; kind: "progress"; stage: "download" | "run"; loaded?: number; total?: number }
  | { id: number; kind: "done"; result: Uint8Array; workSize: string; inferenceMs: number; totalMs: number }
  | { id: number; kind: "error"; message: string };

ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

const sessions = new Map<string, Promise<ort.InferenceSession>>();

async function download(url: string, id: number): Promise<Uint8Array> {
  const res = await fetch(url);
  if (!res.ok || !res.body) throw new Error(`Could not download the model (${res.status} ${res.statusText}).`);
  const total = Number(res.headers.get("content-length")) || undefined;
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    post({ id, kind: "progress", stage: "download", loaded, total });
  }
  const buf = new Uint8Array(loaded);
  let o = 0;
  for (const c of chunks) {
    buf.set(c, o);
    o += c.length;
  }
  return buf;
}

function session(url: string, id: number): Promise<ort.InferenceSession> {
  let s = sessions.get(url);
  if (!s) {
    s = download(url, id).then((bytes) => ort.InferenceSession.create(bytes, { executionProviders: ["wasm"] }));
    s.catch(() => sessions.delete(url));
    sessions.set(url, s);
  }
  return s;
}

function post(msg: WorkerMessage, transfer: Transferable[] = []) {
  self.postMessage(msg, transfer);
}

self.onmessage = async (e: MessageEvent<WorkerRequest>) => {
  const { id, modelUrl, rgb, mask, width: W, height: H, workSize } = e.data;
  try {
    const sess = await session(modelUrl, id);
    post({ id, kind: "progress", stage: "run" });
    const t0 = performance.now();
    const x = toModelInput(rgb, mask, W, H, workSize);
    const t1 = performance.now();
    // Same input convention as training: holes zeroed (done above), mask as a separate input.
    const out = await sess.run({
      masked: new ort.Tensor("float32", x.image, [1, 3, x.hPad, x.wPad]),
      mask: new ort.Tensor("float32", x.mask, [1, 1, x.hPad, x.wPad]),
    });
    const t2 = performance.now();
    const pred = out.prediction.data as Float32Array;
    const result = composite(rgb, predictionToOriginal(pred, x, W, H), mask);
    post({ id, kind: "done", result, workSize: `${x.w}x${x.h}`, inferenceMs: t2 - t1, totalMs: performance.now() - t0 },
      [result.buffer]);
  } catch (err) {
    post({ id, kind: "error", message: (err as Error).message || String(err) });
  }
};
