"""FastAPI backend for the inpainting website.

    GET  /health       liveness and which models are loaded
    GET  /model/info   checkpoints, training resolution, limits, test metrics
    POST /inpaint      multipart: image, mask (white = missing), model, work_size
                       -> image/png (the composite at the original resolution)

Run from the project root:
    .venv/bin/python -m uvicorn backend.app.main:app --port 8000
"""
from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse, Response  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402

from .config import Settings  # noqa: E402
from .engine import Engine, ModelUnavailable  # noqa: E402
from .imaging import InputError, decode_image, decode_mask, encode_png  # noqa: E402

log = logging.getLogger("inpaint.api")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = Engine(settings)
        yield

    app = FastAPI(title="AI Image Inpainting API", version="1.0", lifespan=lifespan)
    app.state.settings = settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", *settings.cors_origins],
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        expose_headers=["X-Model", "X-Work-Size", "X-Hole-Fraction", "X-Inference-Ms", "X-Total-Ms",
                        "X-Image-Size", "X-Had-Alpha"],
    )

    def engine(request: Request) -> Engine:
        return request.app.state.engine

    @app.get("/health")
    def health(request: Request):
        eng = engine(request)
        ok = bool(eng.available)
        return JSONResponse(
            {"status": "ok" if ok else "unavailable", "device": str(eng.device), "models_loaded": eng.available},
            status_code=200 if ok else 503)

    @app.get("/model/info")
    def model_info(request: Request):
        return engine(request).describe()

    @app.post("/inpaint", responses={200: {"content": {"image/png": {}}}})
    async def inpaint(request: Request,
                      image: UploadFile = File(..., description="Photo to repair (PNG, JPEG, WebP, BMP)"),
                      mask: UploadFile = File(..., description="Mask: white = missing, black = known"),
                      model: str | None = Form(None, description="unet or gated_unet"),
                      work_size: int | None = Form(None, description="Longer side (px) the network runs at")):
        eng = engine(request)
        model = model or eng.default_model
        if model is None:
            raise HTTPException(503, "No trained checkpoint is available on the server.")
        if model not in eng.models:
            raise HTTPException(422, f"Unknown model '{model}'. Choose one of: {', '.join(eng.models)}.")
        work_size = work_size or eng.default_work_size(model)
        if not 64 <= work_size <= settings.max_work_size:
            raise HTTPException(422, f"work_size must be between 64 and {settings.max_work_size} px.")

        image_bytes = await _read_limited(image, settings.max_upload_bytes, "image")
        mask_bytes = await _read_limited(mask, settings.max_upload_bytes, "mask")
        try:
            decoded = decode_image(image_bytes, settings.max_pixels)
            m = decode_mask(mask_bytes, decoded.rgb.shape[:2], settings.max_pixels)
        except InputError as exc:
            raise HTTPException(422, str(exc)) from None
        if not m.any():
            raise HTTPException(422, "The mask is empty: paint over the region you want to reconstruct.")
        if m.all():
            raise HTTPException(422, "The mask covers the whole image, so there is no context to inpaint from.")

        try:
            # Sync work (PyTorch + PIL) is run in a worker thread so the event loop stays responsive.
            res = await run_in_threadpool(eng.inpaint, decoded.rgb, m, model, work_size)
            png = await run_in_threadpool(encode_png, res.result)
        except ModelUnavailable as exc:
            raise HTTPException(503, str(exc)) from None
        except Exception:  # noqa: BLE001 - report, never return a fake result
            log.exception("Inference failed")
            raise HTTPException(500, "Inference failed on the server; see the backend log for details.") from None

        H, W = m.shape
        log.info("inpaint model=%s image=%dx%d hole=%.1f%% work=%dx%d infer=%.0fms total=%.0fms",
                 model, W, H, res.hole_fraction * 100, res.work_hw[1], res.work_hw[0], res.inference_ms, res.total_ms)
        return Response(png, media_type="image/png", headers={
            "X-Model": model,
            "X-Work-Size": f"{res.work_hw[1]}x{res.work_hw[0]}",
            "X-Image-Size": f"{W}x{H}",
            "X-Hole-Fraction": f"{res.hole_fraction:.6f}",
            "X-Inference-Ms": f"{res.inference_ms:.1f}",
            "X-Total-Ms": f"{res.total_ms:.1f}",
            "X-Had-Alpha": "1" if decoded.had_alpha else "0",
            "Content-Disposition": f'inline; filename="inpainted_{model}.png"',
            "Cache-Control": "no-store",
        })

    if settings.frontend_dist:
        # Production mode: the built website is served from the same origin as the API.
        app.mount("/", StaticFiles(directory=settings.frontend_dist, html=True), name="frontend")

    return app


async def _read_limited(upload: UploadFile, limit: int, what: str) -> bytes:
    data = await upload.read(limit + 1)
    if len(data) > limit:
        raise HTTPException(413, f"The {what} is larger than {limit / 1024 / 1024:.0f} MB.")
    return data


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
app = create_app()
