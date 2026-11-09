import { afterEach, beforeEach, describe, expect, jest, setSystemTime, test } from "bun:test";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Database } from "bun:sqlite";
import { appendMessage, chatDbPath, closeChatDb, createChat, deleteChat, hasChat, listChats, readChat } from "./chat";

const saved = process.env.STOCKBOT_DATA_DIR;
let directory: string;

beforeEach(async () => {
  directory = await mkdtemp(join(tmpdir(), "chat-store-"));
  process.env.STOCKBOT_DATA_DIR = directory;
  closeChatDb();
});
afterEach(async () => {
  closeChatDb();
  if (saved === undefined) delete process.env.STOCKBOT_DATA_DIR;
  else process.env.STOCKBOT_DATA_DIR = saved;
  await rm(directory, { recursive: true, force: true });
});

describe("chat store", () => {
  test("a new chat takes the first prompt as its title and starts empty", () => {
    const chatId = createChat("What did NVDA file?");
    expect(hasChat(chatId)).toBe(true);
    expect(readChat(chatId)).toMatchObject({ chatId, title: "What did NVDA file?", messages: [] });
  });

  test("messages keep their order and the newest chat lists first", () => {
    const first = createChat("first");
    appendMessage(first, "user", "What did NVDA file?");
    appendMessage(first, "assistant", "An 8-K on 2 September.");
    const second = createChat("second");
    appendMessage(second, "user", "And AAPL?");
    expect(readChat(first)?.messages).toEqual([
      { role: "user", content: "What did NVDA file?" },
      { role: "assistant", content: "An 8-K on 2 September." },
    ]);
    expect(listChats().map((chat) => chat.chatId)).toEqual([second, first]);
  });

  test("deletion removes the chat and its messages, and a missing chat is not deleted twice", () => {
    const chatId = createChat("delete me");
    appendMessage(chatId, "user", "hello");
    expect(deleteChat(chatId)).toBe(true);
    expect(readChat(chatId)).toBeNull();
    expect(listChats()).toEqual([]);
    expect(deleteChat(chatId)).toBe(false);
  });

  test("a later message moves its chat to the top of the list", () => {
    jest.useFakeTimers();
    try {
      // The store stamps each write from the clock, so the clock decides the order.
      setSystemTime(new Date("2026-11-07T10:00:00Z"));
      const older = createChat("older");
      setSystemTime(new Date("2026-11-07T10:00:01Z"));
      const newer = createChat("newer");
      expect(listChats().map((chat) => chat.chatId)).toEqual([newer, older]);
      setSystemTime(new Date("2026-11-07T10:00:02Z"));
      appendMessage(older, "user", "a follow-up question");
      expect(listChats().map((chat) => chat.chatId)).toEqual([older, newer]);
    } finally {
      jest.useRealTimers();
    }
  });

  test("a closed store reopens the same database file", () => {
    const chatId = createChat("kept");
    closeChatDb();
    expect(readChat(chatId)?.chatId).toBe(chatId);
    expect(listChats().map((chat) => chat.chatId)).toEqual([chatId]);
  });

  test("deletion removes the message rows, not only the chat row", () => {
    const chatId = createChat("delete me");
    appendMessage(chatId, "user", "hello");
    expect(deleteChat(chatId)).toBe(true);
    const direct = new Database(chatDbPath(), { readonly: true });
    try {
      expect(direct.query("SELECT COUNT(*) AS rows FROM chat_messages WHERE chat_id = ?").get(chatId)).toEqual({ rows: 0 });
    } finally {
      direct.close();
    }
  });

  test("a long prompt is cut to a readable title", () => {
    const chatId = createChat("x".repeat(200));
    expect(readChat(chatId)?.title).toHaveLength(80);
  });
});
