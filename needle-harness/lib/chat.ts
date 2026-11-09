import { Database } from "bun:sqlite";
import { mkdirSync } from "node:fs";
import { randomUUID } from "node:crypto";
import { dirname, join, resolve } from "node:path";
import type { ChatTurn } from "./agent/types";

// Chat conversations live beside the other data files of this local installation.
// ponytail: one installation is one account — no owner column until real accounts exist.
const ROOT = process.cwd().endsWith("needle-harness") ? join(process.cwd(), "..") : process.cwd();

const SCHEMA = `
CREATE TABLE IF NOT EXISTS chats (
  chat_id TEXT PRIMARY KEY, title TEXT NOT NULL,
  created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS chat_messages (
  chat_id TEXT NOT NULL, sequence INTEGER NOT NULL, role TEXT NOT NULL,
  content TEXT NOT NULL, created_at TEXT NOT NULL,
  PRIMARY KEY (chat_id, sequence));
CREATE INDEX IF NOT EXISTS ix_chats_updated ON chats(updated_at);
`;

export type ChatSummary = { chatId: string; title: string; updatedAt: string };
export type StoredChat = { chatId: string; title: string; updatedAt: string; messages: ChatTurn[] };

// One open handle per database path: tests point STOCKBOT_DATA_DIR at a fresh directory.
const open = new Map<string, Database>();

// Test seam: the file behind the store, for checks that read the tables directly.
export function chatDbPath(): string {
  return resolve(ROOT, process.env.STOCKBOT_DATA_DIR ?? "data", "chat.sqlite");
}

function db(): Database {
  const path = chatDbPath();
  const cached = open.get(path);
  if (cached) return cached;
  mkdirSync(dirname(path), { recursive: true });
  const fresh = new Database(path, { create: true });
  fresh.run("PRAGMA journal_mode = WAL");
  fresh.run(SCHEMA);
  open.set(path, fresh);
  return fresh;
}

// Test hook: drop the cached handles so a removed directory is never reused.
export function closeChatDb(): void {
  for (const handle of open.values()) handle.close();
  open.clear();
}

// The first prompt names the chat; later prompts never rename it.
export function createChat(title: string): string {
  const chatId = randomUUID();
  const at = new Date().toISOString();
  db().run("INSERT INTO chats (chat_id, title, created_at, updated_at) VALUES (?, ?, ?, ?)", [chatId, title.slice(0, 80), at, at]);
  return chatId;
}

export function hasChat(chatId: string): boolean {
  return db().query("SELECT 1 FROM chats WHERE chat_id = ?").get(chatId) !== null;
}

export function appendMessage(chatId: string, role: ChatTurn["role"], content: string): void {
  const handle = db();
  const at = new Date().toISOString();
  const next = handle.query("SELECT COALESCE(MAX(sequence), 0) + 1 AS sequence FROM chat_messages WHERE chat_id = ?").get(chatId) as { sequence: number };
  handle.run("INSERT INTO chat_messages (chat_id, sequence, role, content, created_at) VALUES (?, ?, ?, ?, ?)", [chatId, next.sequence, role, content, at]);
  handle.run("UPDATE chats SET updated_at = ? WHERE chat_id = ?", [at, chatId]);
}

export function listChats(limit = 50): ChatSummary[] {
  // Rowid breaks the tie when two chats share a timestamp: the newer row lists first.
  const rows = db().query("SELECT chat_id, title, updated_at FROM chats ORDER BY updated_at DESC, rowid DESC LIMIT ?").all(limit) as { chat_id: string; title: string; updated_at: string }[];
  return rows.map((row) => ({ chatId: row.chat_id, title: row.title, updatedAt: row.updated_at }));
}

export function readChat(chatId: string): StoredChat | null {
  const handle = db();
  const head = handle.query("SELECT chat_id, title, updated_at FROM chats WHERE chat_id = ?").get(chatId) as { chat_id: string; title: string; updated_at: string } | null;
  if (!head) return null;
  const rows = handle.query("SELECT role, content FROM chat_messages WHERE chat_id = ? ORDER BY sequence").all(chatId) as ChatTurn[];
  return { chatId: head.chat_id, title: head.title, updatedAt: head.updated_at, messages: rows };
}

// The user owns deletion: one call removes the chat and every message of it.
export function deleteChat(chatId: string): boolean {
  const handle = db();
  if (!hasChat(chatId)) return false;
  handle.run("DELETE FROM chat_messages WHERE chat_id = ?", [chatId]);
  handle.run("DELETE FROM chats WHERE chat_id = ?", [chatId]);
  return true;
}
