const BASE = (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, "") ?? "";

export interface TestMetrics {
  n_images: number;
  psnr: number;
  ssim: number;
  hole_psnr: number;
}

export interface ModelEntry {
  name: string;
  label: string;
  available: boolean;
  checkpoint: string;
  error: string | null;
  epoch?: number;
  training_resolution?: number;
  parameters?: number;
  test_metrics?: TestMetrics | null;
}

export interface ModelInfo {
  device: string;
  cuda_device: string | null;
  default_model: string | null;
  default_work_size: number;
  max_work_size: number;
  max_upload_mb: number;
  max_pixels: number;
  models: ModelEntry[];
}

export interface InpaintResponse {
  blob: Blob;
  model: string;
  workSize: string;
  imageSize: string;
  holeFraction: number;
  inferenceMs: number;
  totalMs: number;
}

export class ApiError extends Error {}

async function errorMessage(res: Response): Promise<string> {
  try {
    const body = await res.json();
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) return body.detail.map((d: { msg?: string }) => d.msg).join("; ");
  } catch {
    /* not JSON */
  }
  return `The server returned ${res.status} ${res.statusText}.`;
}

export async function getModelInfo(signal?: AbortSignal): Promise<ModelInfo> {
  let res: Response;
  try {
    res = await fetch(`${BASE}/model/info`, { signal });
  } catch {
    throw new ApiError("Cannot reach the inpainting server. Is the backend running?");
  }
  if (!res.ok) throw new ApiError(await errorMessage(res));
  return res.json();
}

export async function inpaint(
  image: File,
  mask: Blob,
  model: string,
  workSize: number,
  signal?: AbortSignal,
): Promise<InpaintResponse> {
  const form = new FormData();
  form.append("image", image, image.name);
  form.append("mask", mask, "mask.png");
  form.append("model", model);
  form.append("work_size", String(workSize));
  let res: Response;
  try {
    res = await fetch(`${BASE}/inpaint`, { method: "POST", body: form, signal });
  } catch (e) {
    if ((e as Error).name === "AbortError") throw e;
    throw new ApiError("Cannot reach the inpainting server. Is the backend running?");
  }
  if (!res.ok) throw new ApiError(await errorMessage(res));
  const type = res.headers.get("content-type") ?? "";
  if (!type.startsWith("image/png")) throw new ApiError(`Unexpected response type '${type}'.`);
  const h = (k: string) => res.headers.get(k) ?? "";
  return {
    blob: await res.blob(),
    model: h("X-Model"),
    workSize: h("X-Work-Size"),
    imageSize: h("X-Image-Size"),
    holeFraction: Number(h("X-Hole-Fraction")),
    inferenceMs: Number(h("X-Inference-Ms")),
    totalMs: Number(h("X-Total-Ms")),
  };
}
