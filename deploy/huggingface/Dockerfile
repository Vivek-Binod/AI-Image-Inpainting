# Hugging Face Docker Space: CPU-only image serving the inpainting website on port 7860.
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    INPAINT_DEVICE=cpu INPAINT_MAX_WORK_SIZE=256 INPAINT_MAX_UPLOAD_MB=15 OMP_NUM_THREADS=2

COPY requirements-space.txt /tmp/
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r /tmp/requirements-space.txt

# Spaces run the container as uid 1000.
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app
COPY --chown=user . .

EXPOSE 7860
CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
