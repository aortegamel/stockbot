const MAX = 80;

// Readable session title, cut to 80 characters. Thesis alert runs (app/thesis/runner.py
// _prompt_sections) become "<scope> thesis check: <trigger summary>". User questions use their first line.
export function sessionTitle(query: string): string {
  let text = (query.trim().split("\n")[0] ?? "").trim();
  if (query.startsWith("thesis_id:")) {
    const summary = /^trigger summary: (.*)$/m.exec(query)?.[1]?.trim();
    const scope = /"scope": "([^"]+)"/.exec(query)?.[1];
    const head = scope && scope !== "unknown" ? `${scope} thesis check` : "Thesis check";
    text = summary ? `${head}: ${summary}` : head;
  }
  return text.length > MAX ? `${text.slice(0, MAX - 1)}…` : text;
}
