import { createHash } from "node:crypto";
import { existsSync, readFileSync } from "node:fs";

// ponytail: .env is authoritative — process.loadEnvFile never overrides inherited
// exports, so a stale parent env (daemon broker, shell export) silently won. Parse
// here and assign directly so the file always wins; callers log the overrides.

// Minimal dotenv parser: KEY="value", KEY='value', KEY=value, export prefix,
// full-line comments. Quoted values keep inner #; unquoted values cut at " #".
export function parseDotenv(text: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const line of text.split("\n")) {
    const t = line.trim();
    if (!t || t.startsWith("#")) continue;
    const m = t.match(/^(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$/);
    if (!m) continue;
    const raw = m[2].trim();
    if (!raw) {
      out[m[1]] = "";
      continue;
    }
    const q = raw[0];
    if (q === '"' || q === "'") {
      const end = raw.indexOf(q, 1);
      out[m[1]] = end === -1 ? raw.slice(1) : raw.slice(1, end);
      continue;
    }
    const hash = raw.search(/\s#/);
    out[m[1]] = (hash === -1 ? raw : raw.slice(0, hash).trim());
  }
  return out;
}

// Identity without secret material: "<len>:<sha256-12>". Two different keys
// never share it; the key itself never leaves the process.
export function fingerprintKey(key: string): string {
  return `${key.length}:${createHash("sha256").update(key).digest("hex").slice(0, 12)}`;
}

// Local-log redaction only — prefix enough to tell oc_sk_f4… from oc_sk_74….
export function redactKey(key: string): string {
  return key.length <= 11 ? `${key.slice(0, 3)}…` : `${key.slice(0, 11)}…`;
}

export type EnvOverride = { key: string; oldRedacted: string; newRedacted: string };

export function loadDotenvAuthoritative(dotenvPath: string): { loaded: boolean; overridden: EnvOverride[] } {
  if (!existsSync(dotenvPath)) return { loaded: false, overridden: [] };
  const parsed = parseDotenv(readFileSync(dotenvPath, "utf8"));
  const overridden: EnvOverride[] = [];
  for (const [k, v] of Object.entries(parsed)) {
    const old = process.env[k];
    if (old === undefined) {
      process.env[k] = v;
    } else if (old !== v) {
      overridden.push({ key: k, oldRedacted: redactKey(old), newRedacted: redactKey(v) });
      process.env[k] = v;
    }
  }
  return { loaded: true, overridden };
}
