import { useEffect, useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import { clientToMask, displayToMaskLength } from "../lib/geometry";
import type { Point, Size } from "../lib/geometry";
import { drawOp } from "../lib/mask";
import type { MaskOp } from "../lib/mask";

export type Tool = "brush" | "eraser" | "rect";

interface Props {
  imageUrl: string;
  imageSize: Size;
  maskSize: Size;
  committed: HTMLCanvasElement; // rasterised mask, owned by the parent
  version: number; // bumps whenever ``committed`` changes
  tool: Tool;
  brushSize: number; // diameter in displayed CSS px
  showOverlay: boolean;
  disabled: boolean;
  onCommit: (op: MaskOp) => void;
}

export default function MaskEditor(props: Props) {
  const { imageUrl, imageSize, maskSize, committed, version, tool, brushSize, showOverlay, disabled, onCommit } = props;
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const draft = useRef<{ op: MaskOp; pointerId: number } | null>(null);
  const [cursor, setCursor] = useState<{ x: number; y: number } | null>(null);
  const [drawing, setDrawing] = useState(false);

  // Show the committed mask whenever it changes.
  useEffect(() => {
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx) return;
    ctx.save();
    ctx.globalCompositeOperation = "copy";
    ctx.drawImage(committed, 0, 0);
    ctx.restore();
  }, [committed, version, maskSize.width, maskSize.height]);

  const redrawDraft = () => {
    const ctx = canvasRef.current?.getContext("2d");
    if (!ctx || !draft.current) return;
    // Committed state + the in-progress op, drawn with the same routine that commits it.
    ctx.save();
    ctx.globalCompositeOperation = "copy";
    ctx.drawImage(committed, 0, 0);
    ctx.restore();
    drawOp(ctx, draft.current.op, maskSize);
    if (draft.current.op.kind === "rect") {
      const { a, b } = draft.current.op;
      ctx.save();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = Math.max(1, maskSize.width / 400);
      ctx.setLineDash([ctx.lineWidth * 4, ctx.lineWidth * 3]);
      ctx.strokeRect(Math.min(a.x, b.x), Math.min(a.y, b.y), Math.abs(a.x - b.x), Math.abs(a.y - b.y));
      ctx.restore();
    }
  };

  const toMask = (e: { clientX: number; clientY: number }): Point => {
    const box = canvasRef.current!.getBoundingClientRect();
    return clientToMask(e.clientX, e.clientY, box, maskSize);
  };

  const onPointerDown = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    if (disabled || !e.isPrimary || (e.pointerType === "mouse" && e.button !== 0)) return;
    e.preventDefault();
    e.currentTarget.setPointerCapture(e.pointerId);
    const p = toMask(e);
    const box = e.currentTarget.getBoundingClientRect();
    const mode = tool === "eraser" ? "erase" : "paint";
    const op: MaskOp =
      tool === "rect"
        ? { kind: "rect", mode: "paint", a: p, b: p }
        : { kind: "stroke", mode, size: displayToMaskLength(brushSize, box, maskSize), points: [p] };
    draft.current = { op, pointerId: e.pointerId };
    setDrawing(true);
    redrawDraft();
  };

  const onPointerMove = (e: ReactPointerEvent<HTMLCanvasElement>) => {
    if (e.pointerType !== "touch") {
      const box = e.currentTarget.getBoundingClientRect();
      setCursor({ x: e.clientX - box.left, y: e.clientY - box.top });
    }
    const d = draft.current;
    if (!d || d.pointerId !== e.pointerId) return;
    if (d.op.kind === "rect") {
      d.op = { ...d.op, b: toMask(e) };
    } else if (d.op.kind === "stroke") {
      const events = e.nativeEvent.getCoalescedEvents?.() ?? [e.nativeEvent];
      for (const ev of events.length ? events : [e.nativeEvent]) d.op.points.push(toMask(ev));
    }
    redrawDraft();
  };

  const finish = (e: ReactPointerEvent<HTMLCanvasElement>, cancelled = false) => {
    const d = draft.current;
    if (!d || d.pointerId !== e.pointerId) return;
    draft.current = null;
    setDrawing(false);
    const op = d.op;
    const emptyRect = op.kind === "rect" && (Math.abs(op.a.x - op.b.x) < 1 || Math.abs(op.a.y - op.b.y) < 1);
    if (cancelled || emptyRect) {
      const ctx = canvasRef.current?.getContext("2d");
      if (ctx) {
        ctx.save();
        ctx.globalCompositeOperation = "copy";
        ctx.drawImage(committed, 0, 0);
        ctx.restore();
      }
      return;
    }
    onCommit(op);
  };

  const cursorVisible = cursor && tool !== "rect" && !disabled;

  return (
    <div
      className="relative mx-auto select-none overflow-hidden rounded-lg bg-white shadow-sm ring-1 ring-slate-900/10"
      style={{
        aspectRatio: `${imageSize.width} / ${imageSize.height}`,
        width: `min(100%, calc(var(--editor-max-h) * ${imageSize.width / imageSize.height}))`,
      }}
    >
      <img
        src={imageUrl}
        alt="Uploaded photo"
        draggable={false}
        className="pointer-events-none absolute inset-0 h-full w-full"
      />
      <canvas
        ref={canvasRef}
        width={maskSize.width}
        height={maskSize.height}
        data-testid="mask-canvas"
        aria-label="Mask drawing area. Paint over the region to reconstruct."
        className={`absolute inset-0 h-full w-full touch-none transition-opacity ${
          tool === "rect" ? "cursor-crosshair" : "cursor-none"
        } ${disabled ? "cursor-not-allowed" : ""}`}
        style={{ opacity: showOverlay || drawing ? 0.6 : 0 }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={(e) => finish(e)}
        onPointerCancel={(e) => finish(e, true)}
        onPointerLeave={() => setCursor(null)}
        onContextMenu={(e) => e.preventDefault()}
      />
      {cursorVisible && (
        <div
          className="pointer-events-none absolute rounded-full border-2 border-white shadow-[0_0_0_1px_rgba(0,0,0,0.6)]"
          style={{
            width: brushSize,
            height: brushSize,
            left: cursor.x - brushSize / 2,
            top: cursor.y - brushSize / 2,
            background: tool === "eraser" ? "rgba(255,255,255,0.25)" : "rgba(255,45,85,0.25)",
          }}
        />
      )}
    </div>
  );
}
