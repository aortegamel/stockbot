// Copies packaged PDF.js runtime assets into ignored public/pdfjs (same-origin, no CDN).
// Next serves public/ now and Vite will serve it after migration, so both frameworks resolve /pdfjs/*.
import { cpSync, mkdirSync, rmSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const source = join(root, "node_modules", "pdfjs-dist");
const target = join(root, "public", "pdfjs");
rmSync(target, { recursive: true, force: true });
mkdirSync(target, { recursive: true });
for (const asset of ["pdf.worker.min.mjs", "pdf.sandbox.min.mjs"]) cpSync(join(source, "build", asset), join(target, asset));
for (const asset of ["cmaps", "standard_fonts", "wasm", "iccs"]) cpSync(join(source, asset), join(target, asset), { recursive: true });
