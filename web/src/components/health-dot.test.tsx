import { describe, expect, test } from "bun:test";
import { renderToStaticMarkup } from "react-dom/server";
import { HealthDot, healthStatus } from "./health-dot";

describe("healthStatus", () => {
  test("down when the service fails or a required dependency fails", () => {
    expect(healthStatus(null)).toEqual({ tone: "down", failing: ["api"] });
    expect(healthStatus({ ok: false })).toEqual({ tone: "down", failing: ["api"] });
    expect(healthStatus({ ok: true, deps: { jev: { ok: false, required: true }, exa: { ok: false, required: false } } }))
      .toEqual({ tone: "down", failing: ["jev", "exa"] });
  });

  test("warn when only optional dependencies fail, ok otherwise", () => {
    expect(healthStatus({ ok: true, deps: { sec: { ok: true, required: true }, exa: { ok: false, required: false } } }))
      .toEqual({ tone: "warn", failing: ["exa"] });
    expect(healthStatus({ ok: true, deps: { sec: { ok: true, required: true } } })).toEqual({ tone: "ok", failing: [] });
    expect(healthStatus({ ok: true })).toEqual({ tone: "ok", failing: [] });
  });

  test("a malformed dependency entry counts as a required failure", () => {
    expect(healthStatus({ ok: true, deps: { venv: true } })).toEqual({ tone: "down", failing: ["venv"] });
  });
});

test("HealthDot renders a checking status before the fetch settles", () => {
  expect(renderToStaticMarkup(<HealthDot />)).toContain('aria-label="health checking"');
});
