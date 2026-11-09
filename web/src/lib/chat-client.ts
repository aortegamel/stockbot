// Stored conversations of this installation. The server owns the messages; the panel reads and deletes.
import type { ChatTurn } from "../../../needle-harness/lib/agent/types";

export type ChatSummary = { chatId: string; title: string; updatedAt: string };
export type StoredChat = { chatId: string; title: string; updatedAt: string; messages: ChatTurn[] };

async function readJson<T>(res: Response, what: string): Promise<T> {
  const body = (await res.json()) as T & { error?: string };
  if (!res.ok) throw new Error(typeof body.error === "string" ? body.error : `${what} ${res.status}`);
  return body;
}

export async function listChats(signal?: AbortSignal): Promise<ChatSummary[]> {
  const body = await readJson<{ chats?: ChatSummary[] }>(await fetch("/api/chat", { signal, headers: { Accept: "application/json" } }), "chats");
  return body.chats ?? [];
}

export async function fetchChat(chatId: string, signal?: AbortSignal): Promise<StoredChat> {
  return readJson<StoredChat>(await fetch(`/api/chat?id=${encodeURIComponent(chatId)}`, { signal, headers: { Accept: "application/json" } }), "chat");
}

export async function deleteChat(chatId: string): Promise<void> {
  await readJson<{ deleted?: string }>(await fetch(`/api/chat?id=${encodeURIComponent(chatId)}`, { method: "DELETE", headers: { Accept: "application/json" } }), "delete chat");
}
