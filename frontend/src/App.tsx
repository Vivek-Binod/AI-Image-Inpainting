import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ChangeEvent, DragEvent } from "react";
import {
  AlertTriangle,
  Brush,
  CheckCircle2,
  Columns2,
  Download,
  Eraser,
  Eye,
  EyeOff,
  ImagePlus,
  Loader2,
  Pencil,
  Redo2,
  RotateCcw,
  Sparkles,
  Square,
  SplitSquareHorizontal,
  Trash2,
  Undo2,
  Upload,
} from "lucide-react";
import MaskEditor from "./components/MaskEditor";
import type { Tool } from "./components/MaskEditor";
import MaskPreview from "./components/MaskPreview";
import CompareSlider from "./components/CompareSlider";
import { ApiError, getModelInfo as getServerModelInfo, inpaint } from "./lib/api";
import { getModelInfo as getBrowserModelInfo, inpaintInBrowser } from "./lib/browserEngine";
import type { Progress } from "./lib/browserEngine";
import type { InpaintResponse, ModelInfo } from "./lib/api";
import { maskCanvasSize, modelSize } from "./lib/geometry";
import type { Size } from "./lib/geometry";
import { applyOp, countHolePixels, exportMaskPng, renderOps } from "./lib/mask";
import type { MaskOp } from "./lib/mask";

// Static build (VITE_ENGINE=browser): the networks run in the visitor's browser instead of on a server.
const BROWSER = import.meta.env.VITE_ENGINE === "browser";
const getModelInfo = BROWSER ? getBrowserModelInfo : getServerModelInfo;

const ACCEPT = "image/png,image/jpeg,image/webp,image/bmp,image/gif";
const WORK_SIZES = [128, 192, 256, 384, 512];

interface Loaded {
  file: File;
  url: string;
  el: HTMLImageElement;
  size: Size;
  maskSize: Size;
}

interface Result extends InpaintResponse {
  url: string;
  maskedHoleFraction: number;
}

type Status = { kind: "idle" } | { kind: "running"; progress?: Progress } | { kind: "error"; message: string } | { kind: "done" };

export default function App() {
  const [info, setInfo] = useState<ModelInfo | null>(null);
  const [infoError, setInfoError] = useState<string | null>(null);
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [tool, setTool] = useState<Tool>("brush");
  const [brushSize, setBrushSize] = useState(36);
  const [showOverlay, setShowOverlay] = useState(true);
  const [ops, setOps] = useState<MaskOp[]>([]);
  const [redo, setRedo] = useState<MaskOp[]>([]);
  const [version, setVersion] = useState(0);
  const [holePixels, setHolePixels] = useState(0);
  const [model, setModel] = useState<string>("");
  const [workSize, setWorkSize] = useState(128);
  const [status, setStatus] = useState<Status>({ kind: "idle" });
  const [result, setResult] = useState<Result | null>(null);
  const [resultVersion, setResultVersion] = useState(-1);
  const [view, setView] = useState<"edit" | "result">("edit");
  const [compareMode, setCompareMode] = useState<"slider" | "side">("slider");
  const committed = useRef<HTMLCanvasElement>(document.createElement("canvas"));
  const abort = useRef<AbortController | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const workspace = useRef<HTMLElement>(null);

  const loadInfo = useCallback(() => {
    setInfoError(null);
    getModelInfo()
      .then((i) => {
        setInfo(i);
        setModel((m) => (m && i.models.some((x) => x.name === m && x.available) ? m : (i.default_model ?? "")));
        setWorkSize(i.default_work_size);
      })
      .catch((e: Error) => setInfoError(e.message));
  }, []);
  useEffect(loadInfo, [loadInfo]);

  // ----- image loading -----
  const openFile = useCallback(
    (file: File | undefined) => {
      if (!file) return;
      setUploadError(null);
      if (!file.type.startsWith("image/")) {
        setUploadError(`"${file.name}" is not an image. Use PNG, JPEG, WebP or BMP.`);
        return;
      }
      const maxBytes = (info?.max_upload_mb ?? 25) * 1024 * 1024;
      if (file.size > maxBytes) {
        setUploadError(`The file is ${(file.size / 1048576).toFixed(1)} MB; the limit is ${info?.max_upload_mb ?? 25} MB.`);
        return;
      }
      const url = URL.createObjectURL(file);
      const el = new Image();
      el.onload = () => {
        // naturalWidth/Height already include EXIF orientation, as on the server.
        const size = { width: el.naturalWidth, height: el.naturalHeight };
        if (size.width * size.height > (info?.max_pixels ?? 40e6)) {
          URL.revokeObjectURL(url);
          setUploadError(`The image is ${size.width}×${size.height}; the limit is ${((info?.max_pixels ?? 40e6) / 1e6).toFixed(0)} MP.`);
          return;
        }
        if (Math.min(size.width, size.height) < 16) {
          URL.revokeObjectURL(url);
          setUploadError("The image is too small (both sides must be at least 16 px).");
          return;
        }
        const maskSize = maskCanvasSize(size);
        const c = committed.current;
        c.width = maskSize.width;
        c.height = maskSize.height;
        setLoaded((prev) => {
          if (prev) URL.revokeObjectURL(prev.url);
          return { file, url, el, size, maskSize };
        });
        setOps([]);
        setRedo([]);
        setHolePixels(0);
        setVersion((v) => v + 1);
        clearResult();
        setStatus({ kind: "idle" });
        setView("edit");
      };
      el.onerror = () => {
        URL.revokeObjectURL(url);
        setUploadError(`Your browser cannot display "${file.name}". Use PNG, JPEG, WebP or BMP.`);
      };
      el.src = url;
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [info],
  );

  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const f = Array.from(e.clipboardData?.files ?? []).find((x) => x.type.startsWith("image/"));
      if (f) openFile(f);
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [openFile]);

  // ----- mask history -----
  const ctx2d = () => committed.current.getContext("2d", { willReadFrequently: true })!;
  const afterChange = () => {
    setVersion((v) => v + 1);
    setHolePixels(countHolePixels(committed.current));
  };
  const commit = (op: MaskOp) => {
    if (!loaded) return;
    applyOp(ctx2d(), op, loaded.maskSize);
    setOps((o) => [...o, op]);
    setRedo([]);
    afterChange();
  };
  const replay = (next: MaskOp[]) => {
    if (!loaded) return;
    renderOps(ctx2d(), next, loaded.maskSize);
    afterChange();
  };
  const undo = () => {
    if (!ops.length) return;
    const next = ops.slice(0, -1);
    setRedo((r) => [...r, ops[ops.length - 1]]);
    setOps(next);
    replay(next);
  };
  const redoOp = () => {
    if (!redo.length || !loaded) return;
    const op = redo[redo.length - 1];
    setRedo((r) => r.slice(0, -1));
    setOps((o) => [...o, op]);
    applyOp(ctx2d(), op, loaded.maskSize);
    afterChange();
  };
  const clearMask = () => {
    if (holePixels === 0) return;
    commit({ kind: "clear" });
  };

  const running = status.kind === "running";

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!loaded || running || (e.target as HTMLElement).closest("input, select, textarea")) return;
      const k = e.key.toLowerCase();
      if ((e.ctrlKey || e.metaKey) && k === "z") {
        e.preventDefault();
        if (e.shiftKey) redoOp();
        else undo();
      } else if ((e.ctrlKey || e.metaKey) && k === "y") {
        e.preventDefault();
        redoOp();
      } else if (!e.ctrlKey && !e.metaKey && !e.altKey) {
        if (k === "b") setTool("brush");
        else if (k === "e") setTool("eraser");
        else if (k === "r") setTool("rect");
        else if (k === "[") setBrushSize((s) => Math.max(4, s - 4));
        else if (k === "]") setBrushSize((s) => Math.min(160, s + 4));
        else if (k === "h") setShowOverlay((s) => !s);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [loaded, running, redoOp, undo]);

  // ----- inference -----
  function clearResult() {
    setResult((r) => {
      if (r) URL.revokeObjectURL(r.url);
      return null;
    });
    setResultVersion(-1);
  }

  const runInpaint = async () => {
    if (!loaded || !model || holePixels === 0) return;
    abort.current?.abort();
    const ctrl = new AbortController();
    abort.current = ctrl;
    setStatus({ kind: "running" });
    try {
      const { blob: maskBlob, holeFraction } = await exportMaskPng(committed.current);
      const res = BROWSER
        ? await inpaintInBrowser(loaded.el, committed.current, model, workSize, ctrl.signal, (progress) => {
            if (!ctrl.signal.aborted) setStatus({ kind: "running", progress });
          })
        : await inpaint(loaded.file, maskBlob, model, workSize, ctrl.signal);
      clearResult();
      setResult({ ...res, url: URL.createObjectURL(res.blob), maskedHoleFraction: holeFraction });
      setResultVersion(version);
      setStatus({ kind: "done" });
      setView("result");
      // On narrow screens the controls are below the image; bring the result into view.
      requestAnimationFrame(() => {
        const top = workspace.current?.getBoundingClientRect().top ?? 0;
        if (top < 0) workspace.current?.scrollIntoView({ behavior: "smooth", block: "start" });
      });
    } catch (e) {
      if ((e as Error).name === "AbortError") {
        setStatus({ kind: "idle" });
        return;
      }
      setStatus({ kind: "error", message: e instanceof ApiError ? e.message : `Unexpected error: ${(e as Error).message}` });
    } finally {
      if (abort.current === ctrl) abort.current = null;
    }
  };

  const reset = () => {
    abort.current?.abort();
    if (loaded) URL.revokeObjectURL(loaded.url);
    clearResult();
    setLoaded(null);
    setOps([]);
    setRedo([]);
    setHolePixels(0);
    setStatus({ kind: "idle" });
    setView("edit");
    setUploadError(null);
    if (fileInput.current) fileInput.current.value = "";
  };

  const availableModels = info?.models.filter((m) => m.available) ?? [];
  const selected = info?.models.find((m) => m.name === model);
  const holeFraction = loaded ? holePixels / (loaded.maskSize.width * loaded.maskSize.height) : 0;
  const netSize = loaded ? modelSize(loaded.size, workSize) : null;
  const trainedRes = selected?.training_resolution ?? 128;
  const stale = result !== null && resultVersion !== version;
  const downloadName = useMemo(() => {
    if (!loaded || !result) return "inpainted.png";
    const stem = loaded.file.name.replace(/\.[^.]+$/, "") || "image";
    return `${stem}_inpainted_${result.model}.png`;
  }, [loaded, result]);

  const inpaintDisabledReason = !loaded
    ? "Upload an image first."
    : holePixels === 0
      ? "Paint over the region you want to reconstruct."
      : holeFraction >= 0.999
        ? "The mask covers the whole image."
        : !model
          ? BROWSER ? "No trained model is available." : "No trained model is available on the server."
          : null;

  return (
    <div className="min-h-dvh bg-slate-50 text-slate-900">
      <header className="sticky top-0 z-20 border-b border-slate-200 bg-white/90 backdrop-blur">
        <div className="mx-auto flex max-w-7xl items-center gap-3 px-4 py-3">
          <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-blue-700 text-white">
            <Sparkles className="h-5 w-5" aria-hidden />
          </div>
          <div className="min-w-0">
            <h1 className="truncate text-base font-semibold sm:text-lg">AI Image Inpainting</h1>
            <p className="hidden truncate text-xs text-slate-500 sm:block">Restoring missing visual content with U-Net models</p>
          </div>
          <div className="ml-auto flex items-center gap-2">
            <ServerPill info={info} error={infoError} onRetry={loadInfo} />
            {loaded && (
              <button onClick={reset} className="btn-secondary" data-testid="reset">
                <RotateCcw className="h-4 w-4" aria-hidden />
                <span className="hidden sm:inline">New image</span>
              </button>
            )}
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-4 py-5">
        {!loaded ? (
          <UploadArea
            onFile={openFile}
            error={uploadError}
            inputRef={fileInput}
            info={info}
          />
        ) : (
          <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_340px]">
            {/* ---------- workspace ---------- */}
            <section ref={workspace} className="min-w-0 scroll-mt-20">
              <div className="mb-3 flex flex-wrap items-center gap-2">
                <div className="inline-flex rounded-lg bg-slate-200/70 p-1" role="tablist">
                  <TabButton active={view === "edit"} onClick={() => setView("edit")} icon={<Pencil className="h-4 w-4" />}>
                    Edit mask
                  </TabButton>
                  <TabButton
                    active={view === "result"}
                    onClick={() => setView("result")}
                    disabled={!result}
                    icon={<Sparkles className="h-4 w-4" />}
                  >
                    Result
                  </TabButton>
                </div>
                <p className="truncate text-xs text-slate-500">
                  {loaded.file.name} · {loaded.size.width}×{loaded.size.height}px
                </p>
              </div>

              {view === "edit" ? (
                <>
                  <Toolbar
                    tool={tool}
                    setTool={setTool}
                    brushSize={brushSize}
                    setBrushSize={setBrushSize}
                    showOverlay={showOverlay}
                    setShowOverlay={setShowOverlay}
                    canUndo={ops.length > 0}
                    canRedo={redo.length > 0}
                    canClear={holePixels > 0}
                    onUndo={undo}
                    onRedo={redoOp}
                    onClear={clearMask}
                    disabled={running}
                  />
                  <MaskEditor
                    imageUrl={loaded.url}
                    imageSize={loaded.size}
                    maskSize={loaded.maskSize}
                    committed={committed.current}
                    version={version}
                    tool={tool}
                    brushSize={brushSize}
                    showOverlay={showOverlay}
                    disabled={running}
                    onCommit={commit}
                  />
                  <p className="mt-2 text-center text-xs text-slate-500">
                    {tool === "rect" ? "Drag to add a rectangular hole." : tool === "eraser" ? "Paint to remove parts of the mask." : "Paint over what you want removed and reconstructed."}{" "}
                    Red pixels will be reconstructed; everything else is kept exactly.
                  </p>
                </>
              ) : (
                result && (
                  <ResultView
                    loaded={loaded}
                    result={result}
                    stale={stale}
                    compareMode={compareMode}
                    setCompareMode={setCompareMode}
                    downloadName={downloadName}
                    onEdit={() => setView("edit")}
                    onReset={reset}
                    committed={committed.current}
                    version={resultVersion}
                  />
                )
              )}
            </section>

            {/* ---------- side panel ---------- */}
            <aside className="space-y-4">
              <Panel title="Masked input preview">
                <MaskPreview image={loaded.el} imageSize={loaded.size} committed={committed.current} version={version} />
                <dl className="mt-2 grid grid-cols-2 gap-x-3 gap-y-1 text-xs">
                  <dt className="text-slate-500">Region to fill</dt>
                  <dd className="text-right font-medium tabular-nums" data-testid="hole-fraction">
                    {(holeFraction * 100).toFixed(1)}%
                  </dd>
                  <dt className="text-slate-500">Network input</dt>
                  <dd className="text-right font-medium tabular-nums">
                    {netSize ? `${netSize.width}×${netSize.height}px` : "–"}
                  </dd>
                </dl>
                {holeFraction > 0.4 && (
                  <p className="mt-2 rounded-md bg-amber-50 px-2 py-1.5 text-xs text-amber-800">
                    Large holes give blurry results: the models were trained on holes of up to about half the image.
                  </p>
                )}
              </Panel>

              <Panel title="Model">
                {info ? (
                  <div className="space-y-2" role="radiogroup" aria-label="Model">
                    {info.models.map((m) => (
                      <label
                        key={m.name}
                        className={`flex cursor-pointer items-start gap-3 rounded-lg border p-3 text-sm transition ${
                          !m.available
                            ? "cursor-not-allowed border-slate-200 opacity-60"
                            : model === m.name
                              ? "border-blue-600 bg-blue-50/60 ring-1 ring-blue-600"
                              : "border-slate-200 hover:border-slate-300"
                        }`}
                      >
                        <input
                          type="radio"
                          name="model"
                          value={m.name}
                          checked={model === m.name}
                          disabled={!m.available || running}
                          onChange={() => setModel(m.name)}
                          className="mt-0.5 accent-blue-700"
                          data-testid={`model-${m.name}`}
                        />
                        <span className="min-w-0">
                          <span className="block font-medium">{m.label}</span>
                          {m.available ? (
                            <span className="block text-xs text-slate-500">
                              {((m.parameters ?? 0) / 1e6).toFixed(1)}M params · {m.epoch} epochs
                              {m.test_metrics && <> · test hole PSNR {m.test_metrics.hole_psnr.toFixed(2)} dB</>}
                            </span>
                          ) : (
                            <span className="block text-xs break-words text-red-700">Unavailable: {m.error}</span>
                          )}
                        </span>
                      </label>
                    ))}
                    {availableModels.length === 0 && (
                      <p className="text-xs text-red-700">No trained checkpoint is available. Inpainting is disabled.</p>
                    )}
                  </div>
                ) : (
                  <p className="text-sm text-slate-500">{infoError ? "Server unavailable." : "Loading models…"}</p>
                )}

                <details className="mt-3 text-sm">
                  <summary className="cursor-pointer text-slate-600 select-none">Processing resolution</summary>
                  <div className="mt-2 space-y-2">
                    <select
                      value={workSize}
                      onChange={(e) => setWorkSize(Number(e.target.value))}
                      disabled={running}
                      className="w-full rounded-md border border-slate-300 bg-white px-2 py-1.5 text-sm"
                      aria-label="Processing resolution"
                    >
                      {WORK_SIZES.filter((s) => s <= (info?.max_work_size ?? 512)).map((s) => (
                        <option key={s} value={s}>
                          Longer side {s}px{s === trainedRes ? " (training resolution)" : ""}
                        </option>
                      ))}
                    </select>
                    <p className="text-xs text-slate-500">
                      The networks were trained on {trainedRes}×{trainedRes} images. The hole is filled at this
                      resolution and upscaled, so fine detail is not recovered. On 256 px test photos, running at 256
                      gave similar PSNR and slightly better LPIPS than 128; larger values are untested. Known pixels
                      are always copied from your original at full resolution.
                    </p>
                  </div>
                </details>
              </Panel>

              <div className="space-y-2">
                <button
                  onClick={runInpaint}
                  disabled={!!inpaintDisabledReason || running}
                  className="btn-primary w-full py-3 text-base"
                  data-testid="inpaint"
                >
                  {running ? <Loader2 className="h-5 w-5 animate-spin" aria-hidden /> : <Sparkles className="h-5 w-5" aria-hidden />}
                  {running ? runningLabel(status, selected?.label ?? "model") : result ? "Inpaint again" : "Inpaint"}
                </button>
                {running && (
                  <button onClick={() => abort.current?.abort()} className="btn-secondary w-full">
                    Cancel
                  </button>
                )}
                {inpaintDisabledReason && !running && <p className="text-center text-xs text-slate-500">{inpaintDisabledReason}</p>}
                {status.kind === "error" && (
                  <div role="alert" className="flex gap-2 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-800" data-testid="error">
                    <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
                    <span className="break-words">{status.message}</span>
                  </div>
                )}
                {status.kind === "done" && result && (
                  <div role="status" className="flex gap-2 rounded-lg border border-emerald-200 bg-emerald-50 p-3 text-sm text-emerald-800">
                    <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0" aria-hidden />
                    <span>
                      Done with {info?.models.find((m) => m.name === result.model)?.label ?? result.model} in{" "}
                      {(result.totalMs / 1000).toFixed(2)} s (network {result.inferenceMs.toFixed(0)} ms at {result.workSize}).
                    </span>
                  </div>
                )}
              </div>
            </aside>
          </div>
        )}
      </main>

      <footer className="mx-auto max-w-7xl px-4 pt-2 pb-8 text-xs text-slate-500">
        Results are generated by the selected network and are a plausible estimate, not the true hidden content. The
        models work at {info?.models.find((m) => m.available)?.training_resolution ?? 128}px, so filled regions are
        blurry, especially in large holes.
      </footer>
    </div>
  );
}

// ---------------------------------------------------------------------------

function runningLabel(status: Status, label: string): string {
  const p = status.kind === "running" ? status.progress : undefined;
  if (p?.stage === "download") {
    const mb = (b?: number) => ((b ?? 0) / 1048576).toFixed(0);
    return p.total ? `Downloading ${label}… ${mb(p.loaded)}/${mb(p.total)} MB` : `Downloading ${label}…`;
  }
  return `Inpainting with ${label}…`;
}

function ServerPill({ info, error, onRetry }: { info: ModelInfo | null; error: string | null; onRetry: () => void }) {
  if (error)
    return (
      <button onClick={onRetry} className="inline-flex items-center gap-1.5 rounded-full bg-red-50 px-2.5 py-1 text-xs font-medium text-red-700 ring-1 ring-red-200" title={error}>
        <span className="h-2 w-2 rounded-full bg-red-500" /> <span className="whitespace-nowrap">Offline · retry</span>
      </button>
    );
  if (!info)
    return <span className="rounded-full bg-slate-100 px-2.5 py-1 text-xs text-slate-500">Connecting…</span>;
  const n = info.models.filter((m) => m.available).length;
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium whitespace-nowrap ring-1 ${
        n ? "bg-emerald-50 text-emerald-700 ring-emerald-200" : "bg-amber-50 text-amber-800 ring-amber-200"
      }`}
      title={info.cuda_device ?? info.device}
      data-testid="server-status"
    >
      <span className={`h-2 w-2 rounded-full ${n ? "bg-emerald-500" : "bg-amber-500"}`} />
      {n} model{n === 1 ? "" : "s"} · {info.device === "browser" ? "in browser" : info.device.startsWith("cuda") ? "GPU" : "CPU"}
    </span>
  );
}

function UploadArea({
  onFile,
  error,
  inputRef,
  info,
}: {
  onFile: (f: File | undefined) => void;
  error: string | null;
  inputRef: React.RefObject<HTMLInputElement | null>;
  info: ModelInfo | null;
}) {
  const [over, setOver] = useState(false);
  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setOver(false);
    onFile(e.dataTransfer.files[0]);
  };
  return (
    <div className="mx-auto max-w-3xl">
      <label
        onDragOver={(e) => {
          e.preventDefault();
          setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={onDrop}
        className={`flex cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed px-6 py-16 text-center transition sm:py-24 ${
          over ? "border-blue-600 bg-blue-50" : "border-slate-300 bg-white hover:border-blue-400 hover:bg-blue-50/40"
        }`}
        data-testid="dropzone"
      >
        <div className="mb-4 flex h-14 w-14 items-center justify-center rounded-full bg-blue-100 text-blue-700">
          <ImagePlus className="h-7 w-7" aria-hidden />
        </div>
        <span className="text-lg font-semibold">Upload a photo</span>
        <span className="mt-1 text-sm text-slate-500">Drag and drop, paste, or click to browse · PNG, JPEG, WebP, BMP</span>
        <span className="btn-primary mt-6">
          <Upload className="h-4 w-4" aria-hidden /> Choose image
        </span>
        <input
          ref={inputRef}
          type="file"
          accept={ACCEPT}
          className="sr-only"
          data-testid="file-input"
          onChange={(e: ChangeEvent<HTMLInputElement>) => onFile(e.target.files?.[0])}
        />
      </label>
      {error && (
        <div role="alert" className="mt-4 flex gap-2 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-800" data-testid="upload-error">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" aria-hidden /> {error}
        </div>
      )}
      <ol className="mt-8 grid gap-3 text-sm sm:grid-cols-3">
        {[
          ["1", "Upload", BROWSER ? "Any photo; it never leaves your device." : "Any photo; it never leaves this server."],
          ["2", "Paint the hole", "Brush, eraser or rectangle over what should be removed."],
          ["3", "Inpaint & download", "Compare before/after and save the PNG."],
        ].map(([n, t, d]) => (
          <li key={n} className="rounded-xl bg-white p-4 ring-1 ring-slate-200">
            <span className="text-xs font-semibold text-blue-700">STEP {n}</span>
            <p className="mt-1 font-medium">{t}</p>
            <p className="mt-0.5 text-slate-500">{d}</p>
          </li>
        ))}
      </ol>
      {info && (
        <p className="mt-6 text-center text-xs text-slate-500">
          Models trained on {info.models.find((m) => m.available)?.training_resolution ?? 128}×
          {info.models.find((m) => m.available)?.training_resolution ?? 128} Places365 scenes.{" "}
          {BROWSER
            ? "They run in your browser: the first Inpaint downloads the selected model (17 or 35 MB)."
            : `Running on ${info.cuda_device ?? info.device}.`}
        </p>
      )}
    </div>
  );
}

function Toolbar(p: {
  tool: Tool;
  setTool: (t: Tool) => void;
  brushSize: number;
  setBrushSize: (n: number) => void;
  showOverlay: boolean;
  setShowOverlay: (b: boolean) => void;
  canUndo: boolean;
  canRedo: boolean;
  canClear: boolean;
  onUndo: () => void;
  onRedo: () => void;
  onClear: () => void;
  disabled: boolean;
}) {
  const tools: [Tool, string, React.ReactNode, string][] = [
    ["brush", "Brush", <Brush className="h-4 w-4" />, "B"],
    ["eraser", "Eraser", <Eraser className="h-4 w-4" />, "E"],
    ["rect", "Rectangle", <Square className="h-4 w-4" />, "R"],
  ];
  return (
    <div className="mb-3 flex flex-wrap items-center gap-2 rounded-xl bg-white p-2 shadow-sm ring-1 ring-slate-200" data-testid="toolbar">
      <div className="inline-flex rounded-lg bg-slate-100 p-0.5" role="group" aria-label="Drawing tool">
        {tools.map(([t, label, icon, key]) => (
          <button
            key={t}
            onClick={() => p.setTool(t)}
            aria-pressed={p.tool === t}
            title={`${label} (${key})`}
            className={`inline-flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-sm font-medium transition ${
              p.tool === t ? "bg-white text-blue-700 shadow-sm" : "text-slate-600 hover:text-slate-900"
            }`}
            data-testid={`tool-${t}`}
          >
            {icon}
            <span className="hidden sm:inline">{label}</span>
          </button>
        ))}
      </div>
      <label className={`flex min-w-[150px] flex-1 items-center gap-2 text-sm text-slate-600 ${p.tool === "rect" ? "opacity-40" : ""}`}>
        <span className="shrink-0">Size</span>
        <input
          type="range"
          min={4}
          max={160}
          value={p.brushSize}
          onChange={(e) => p.setBrushSize(Number(e.target.value))}
          disabled={p.tool === "rect"}
          className="w-full accent-blue-700"
          aria-label="Brush size"
          data-testid="brush-size"
        />
        <span className="w-9 shrink-0 text-right text-xs tabular-nums">{p.brushSize}px</span>
      </label>
      <div className="flex items-center gap-1">
        <IconButton label="Undo (Ctrl+Z)" onClick={p.onUndo} disabled={!p.canUndo || p.disabled} testId="undo">
          <Undo2 className="h-4 w-4" />
        </IconButton>
        <IconButton label="Redo (Ctrl+Shift+Z)" onClick={p.onRedo} disabled={!p.canRedo || p.disabled} testId="redo">
          <Redo2 className="h-4 w-4" />
        </IconButton>
        <IconButton
          label={p.showOverlay ? "Hide mask overlay (H)" : "Show mask overlay (H)"}
          onClick={() => p.setShowOverlay(!p.showOverlay)}
          pressed={!p.showOverlay}
          testId="toggle-overlay"
        >
          {p.showOverlay ? <Eye className="h-4 w-4" /> : <EyeOff className="h-4 w-4" />}
        </IconButton>
        <IconButton label="Clear mask" onClick={p.onClear} disabled={!p.canClear || p.disabled} testId="clear">
          <Trash2 className="h-4 w-4" />
        </IconButton>
      </div>
    </div>
  );
}

function ResultView(p: {
  loaded: Loaded;
  result: Result;
  stale: boolean;
  compareMode: "slider" | "side";
  setCompareMode: (m: "slider" | "side") => void;
  downloadName: string;
  onEdit: () => void;
  onReset: () => void;
  committed: HTMLCanvasElement;
  version: number;
}) {
  const { loaded, result } = p;
  return (
    <div data-testid="result-view">
      <div className="mb-3 flex flex-wrap items-center gap-2 rounded-xl bg-white p-2 shadow-sm ring-1 ring-slate-200">
        <div className="inline-flex rounded-lg bg-slate-100 p-0.5" role="group" aria-label="Comparison mode">
          <TabButton active={p.compareMode === "slider"} onClick={() => p.setCompareMode("slider")} icon={<SplitSquareHorizontal className="h-4 w-4" />}>
            Slider
          </TabButton>
          <TabButton active={p.compareMode === "side"} onClick={() => p.setCompareMode("side")} icon={<Columns2 className="h-4 w-4" />}>
            Side by side
          </TabButton>
        </div>
        <div className="ml-auto flex flex-wrap gap-2">
          <button onClick={p.onEdit} className="btn-secondary">
            <Pencil className="h-4 w-4" aria-hidden /> Edit mask
          </button>
          <a href={result.url} download={p.downloadName} className="btn-primary" data-testid="download">
            <Download className="h-4 w-4" aria-hidden /> Download PNG
          </a>
        </div>
      </div>
      {p.stale && (
        <p className="mb-3 rounded-md bg-amber-50 px-3 py-2 text-xs text-amber-800">
          The mask has changed since this result was produced. Click Inpaint again to update it.
        </p>
      )}
      {p.compareMode === "slider" ? (
        <CompareSlider before={loaded.url} after={result.url} size={loaded.size} />
      ) : (
        <div className="grid gap-3 sm:grid-cols-2">
          <figure>
            <img src={loaded.url} alt="Original" className="w-full rounded-lg ring-1 ring-slate-900/10" />
            <figcaption className="mt-1 text-center text-xs text-slate-500">Original</figcaption>
          </figure>
          <figure>
            <img src={result.url} alt="Inpainted result" className="w-full rounded-lg ring-1 ring-slate-900/10" data-testid="result-image" />
            <figcaption className="mt-1 text-center text-xs text-slate-500">
              Inpainted ({result.model === "gated_unet" ? "Gated U-Net" : "U-Net"})
            </figcaption>
          </figure>
        </div>
      )}
      <p className="mt-2 text-center text-xs text-slate-500">
        {result.imageSize} PNG · {(result.holeFraction * 100).toFixed(1)}% of pixels reconstructed at {result.workSize} · all other
        pixels are identical to the original.
      </p>
      <div className="mt-4 flex justify-center">
        <button onClick={p.onReset} className="btn-secondary">
          <RotateCcw className="h-4 w-4" aria-hidden /> Process another image
        </button>
      </div>
    </div>
  );
}

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="rounded-xl bg-white p-4 shadow-sm ring-1 ring-slate-200">
      <h2 className="mb-3 text-sm font-semibold text-slate-700">{title}</h2>
      {children}
    </section>
  );
}

function TabButton(p: { active: boolean; onClick: () => void; disabled?: boolean; icon: React.ReactNode; children: React.ReactNode }) {
  return (
    <button
      role="tab"
      aria-selected={p.active}
      onClick={p.onClick}
      disabled={p.disabled}
      className={`inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm font-medium transition disabled:opacity-40 ${
        p.active ? "bg-white text-slate-900 shadow-sm" : "text-slate-600 hover:text-slate-900"
      }`}
    >
      {p.icon}
      {p.children}
    </button>
  );
}

function IconButton(p: { label: string; onClick: () => void; disabled?: boolean; pressed?: boolean; testId?: string; children: React.ReactNode }) {
  return (
    <button
      onClick={p.onClick}
      disabled={p.disabled}
      title={p.label}
      aria-label={p.label}
      aria-pressed={p.pressed}
      data-testid={p.testId}
      className={`inline-flex h-9 w-9 items-center justify-center rounded-md text-slate-600 transition hover:bg-slate-100 hover:text-slate-900 disabled:pointer-events-none disabled:opacity-35 ${
        p.pressed ? "bg-slate-100 text-slate-900" : ""
      }`}
    >
      {p.children}
    </button>
  );
}
