import { useRef, useState } from "react";
import type { PointerEvent as ReactPointerEvent } from "react";
import type { Size } from "../lib/geometry";

interface Props {
  before: string;
  after: string;
  size: Size;
  beforeLabel?: string;
  afterLabel?: string;
}

/** Before/after comparison: drag the handle (or use the arrow keys) to reveal the result. */
export default function CompareSlider({ before, after, size, beforeLabel = "Original", afterLabel = "Result" }: Props) {
  const [pos, setPos] = useState(50);
  const box = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);

  const update = (clientX: number) => {
    const r = box.current!.getBoundingClientRect();
    setPos(Math.min(100, Math.max(0, ((clientX - r.left) / r.width) * 100)));
  };
  const down = (e: ReactPointerEvent) => {
    dragging.current = true;
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    update(e.clientX);
  };

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowLeft" || e.key === "ArrowDown") {
      e.preventDefault();
      setPos((p) => Math.max(0, p - (e.shiftKey ? 10 : 2)));
    } else if (e.key === "ArrowRight" || e.key === "ArrowUp") {
      e.preventDefault();
      setPos((p) => Math.min(100, p + (e.shiftKey ? 10 : 2)));
    }
  };

  return (
    <div
      ref={box}
      tabIndex={0}
      role="slider"
      aria-label="Comparison slider"
      aria-valuenow={Math.round(pos)}
      aria-valuemin={0}
      aria-valuemax={100}
      onKeyDown={onKeyDown}
      className="relative mx-auto touch-none select-none overflow-hidden rounded-lg bg-white shadow-sm ring-1 ring-slate-900/10 focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500"
      style={{
        aspectRatio: `${size.width} / ${size.height}`,
        width: `min(100%, calc(var(--editor-max-h) * ${size.width / size.height}))`,
      }}
      onPointerDown={down}
      onPointerMove={(e) => dragging.current && update(e.clientX)}
      onPointerUp={() => (dragging.current = false)}
      onPointerCancel={() => (dragging.current = false)}
      data-testid="compare"
    >
      <img src={before} alt={beforeLabel} draggable={false} className="absolute inset-0 h-full w-full" />
      <img
        src={after}
        alt={afterLabel}
        draggable={false}
        className="absolute inset-0 h-full w-full"
        style={{ clipPath: `inset(0 0 0 ${pos}%)` }}
      />
      <div className="pointer-events-none absolute inset-y-0 w-0.5 bg-white shadow-[0_0_0_1px_rgba(0,0,0,0.35)]" style={{ left: `${pos}%` }}>
        <div className="absolute top-1/2 left-1/2 flex h-9 w-9 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full bg-white text-slate-700 shadow-md ring-1 ring-slate-900/15">
          <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M9 6l-6 6 6 6M15 6l6 6-6 6" />
          </svg>
        </div>
      </div>
      <span className="pointer-events-none absolute top-2 left-2 rounded bg-slate-900/70 px-2 py-0.5 text-xs font-medium text-white">
        {beforeLabel}
      </span>
      <span className="pointer-events-none absolute top-2 right-2 rounded bg-slate-900/70 px-2 py-0.5 text-xs font-medium text-white">
        {afterLabel}
      </span>
      <input
        type="range"
        min={0}
        max={100}
        step={0.5}
        value={pos}
        onChange={(e) => setPos(Number(e.target.value))}
        aria-label="Comparison position"
        className="sr-only"
      />
    </div>
  );
}
