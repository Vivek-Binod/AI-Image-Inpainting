import { useEffect, useRef } from "react";
import type { Size } from "../lib/geometry";

interface Props {
  image: HTMLImageElement | null;
  imageSize: Size;
  committed: HTMLCanvasElement;
  version: number;
}

const PREVIEW_MAX = 720;

/** The image as the network receives it: masked pixels removed (shown as a checkerboard). */
export default function MaskPreview({ image, imageSize, committed, version }: Props) {
  const ref = useRef<HTMLCanvasElement>(null);
  const scale = Math.min(1, PREVIEW_MAX / Math.max(imageSize.width, imageSize.height));
  const w = Math.max(1, Math.round(imageSize.width * scale));
  const h = Math.max(1, Math.round(imageSize.height * scale));

  useEffect(() => {
    const canvas = ref.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx || !image) return;
    ctx.fillStyle = "#fff";
    ctx.fillRect(0, 0, w, h);
    ctx.drawImage(image, 0, 0, w, h);

    // Hole layer: checkerboard clipped to the mask.
    const hole = document.createElement("canvas");
    hole.width = w;
    hole.height = h;
    const hctx = hole.getContext("2d")!;
    hctx.imageSmoothingEnabled = false;
    hctx.drawImage(committed, 0, 0, w, h);
    hctx.globalCompositeOperation = "source-in";
    const tile = document.createElement("canvas");
    tile.width = tile.height = 12;
    const tctx = tile.getContext("2d")!;
    tctx.fillStyle = "#cbd5e1";
    tctx.fillRect(0, 0, 12, 12);
    tctx.fillStyle = "#94a3b8";
    tctx.fillRect(0, 0, 6, 6);
    tctx.fillRect(6, 6, 6, 6);
    hctx.fillStyle = hctx.createPattern(tile, "repeat")!;
    hctx.fillRect(0, 0, w, h);
    ctx.drawImage(hole, 0, 0);
  }, [image, committed, version, w, h]);

  return (
    <canvas
      ref={ref}
      width={w}
      height={h}
      data-testid="mask-preview"
      className="h-auto w-full rounded-md ring-1 ring-slate-900/10"
      aria-label="Preview of the image with the masked region removed"
    />
  );
}
