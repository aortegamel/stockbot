"use client";

import { useState } from "react";

export function PromptBox({ onSubmit, busy }: { onSubmit: (prompt: string) => void; busy: boolean }) {
  const [value, setValue] = useState("");
  return (
    <form
      className="flex gap-2 border-t border-zinc-800 p-3"
      onSubmit={(e) => {
        e.preventDefault();
        const prompt = value.trim();
        if (!prompt || busy) return;
        setValue("");
        onSubmit(prompt);
      }}
    >
      <span className="pt-2 text-green-500">&gt;</span>
      <input
        className="flex-1 bg-transparent text-sm text-zinc-100 outline-none placeholder:text-zinc-600"
        placeholder="Ask Stockbot..."
        value={value}
        onChange={(e) => setValue(e.target.value)}
        disabled={busy}
      />
      <button
        type="submit"
        disabled={busy || !value.trim()}
        className="rounded border border-zinc-700 px-3 py-1 text-sm text-zinc-300 disabled:opacity-40"
      >
        SEND
      </button>
    </form>
  );
}
