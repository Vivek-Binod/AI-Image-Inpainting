// Copies the ONNX models and models.json (from ../build/onnx, written by scripts/export_onnx.py)
// into the static build so the browser can download them from ./models/.
import { cpSync, existsSync, mkdirSync, readdirSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const src = path.resolve(here, "../../build/onnx");
const dest = path.resolve(here, "..", process.argv[2] ?? "dist-static", "models");
const files = ["models.json", "unet.onnx", "gated_unet.onnx"];
for (const f of files) {
  if (!existsSync(path.join(src, f))) {
    console.error(`Missing ${path.join(src, f)}. Run: .venv/bin/python scripts/export_onnx.py (from the project root)`);
    process.exit(1);
  }
}
mkdirSync(dest, { recursive: true });
for (const f of files) cpSync(path.join(src, f), path.join(dest, f));
console.log(`copied ${readdirSync(dest).map((f) => `${f} (${(statSync(path.join(dest, f)).size / 1e6).toFixed(1)} MB)`).join(", ")} -> ${dest}`);
