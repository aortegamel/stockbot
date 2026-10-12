// Copies packaged PDF.js runtime assets into web/public/pdfjs for same-origin loading.
import { cpSync, mkdirSync, rmSync } from "node:fs";
import { createRequire } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "web");
const source = dirname(createRequire(join(root, "package.json")).resolve("pdfjs-dist/package.json"));
const target = join(root, "public", "pdfjs");
rmSync(target, { recursive: true, force: true });
mkdirSync(target, { recursive: true });
for (const asset of ["pdf.worker.min.mjs", "pdf.sandbox.min.mjs"]) cpSync(join(source, "build", asset), join(target, asset));
for (const asset of ["cmaps", "standard_fonts", "wasm", "iccs"]) cpSync(join(source, asset), join(target, asset), { recursive: true });
