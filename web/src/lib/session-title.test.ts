import { describe, expect, test } from "bun:test";
import { sessionTitle } from "./session-title";
import { ago } from "./time";

// Shape of app/thesis/runner.py _prompt_sections output.
const thesisPrompt = (summary: string, scope = "NVDA") =>
  [
    "thesis_id: thesis:6c0f2975",
    "trigger_id: trigger:040bf131",
    "run_id: run:20261020T191326905359",
    "TRIGGER DATA CUTOFF: 2026-10-20T19:13:26+00:00",
    `trigger summary: ${summary}`,
    "trigger canonical refs: sec:0001199039-26-000016",
    "CURRENT THESIS STATE:",
    `{"memories": [], "thesis": {"scope": "${scope}", "slug": "nvda"}}`,
  ].join("\n");

describe("sessionTitle", () => {
  test("turns a thesis alert prompt into a scoped short title", () => {
    expect(sessionTitle(thesisPrompt("short-interest settlement 2026-10-15 for NVDA"))).toBe(
      "NVDA thesis check: short-interest settlement 2026-10-15 for NVDA",
    );
    expect(sessionTitle(thesisPrompt("4 filed", "unknown"))).toBe("Thesis check: 4 filed");
    expect(sessionTitle("thesis_id: thesis:x\ntrigger summary: ")).toBe("Thesis check");
    expect(sessionTitle(thesisPrompt("x".repeat(200)))).toHaveLength(80);
  });

  test("uses the first line of a user question, cut to 80 characters", () => {
    expect(sessionTitle("  What do you think about the CEO?\nmore context")).toBe("What do you think about the CEO?");
    const long = sessionTitle("Q".repeat(120));
    expect(long).toHaveLength(80);
    expect(long.endsWith("…")).toBe(true);
    expect(sessionTitle("Q".repeat(80))).toBe("Q".repeat(80));
    expect(sessionTitle("")).toBe("");
  });
});

describe("ago", () => {
  test("formats elapsed time in the largest whole unit", () => {
    const now = Date.parse("2026-10-20T12:00:00Z");
    expect(ago("2026-10-20T11:59:30Z", now)).toBe("30s");
    expect(ago("2026-10-20T11:58:00Z", now)).toBe("2m");
    expect(ago("2026-10-20T10:00:00Z", now)).toBe("2h");
    expect(ago("2026-10-18T12:00:00Z", now)).toBe("2d");
    expect(ago("2026-10-20T12:00:05Z", now)).toBe("0s");
    expect(ago("not a date", now)).toBe("");
  });
});
