import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { mkdtemp, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { appendMessage, closeChatDb, createChat } from "@/lib/chat";
import { DELETE, GET } from "./route";

const saved = process.env.STOCKBOT_DATA_DIR;
let directory: string;

beforeEach(async () => {
  directory = await mkdtemp(join(tmpdir(), "chat-route-"));
  process.env.STOCKBOT_DATA_DIR = directory;
  closeChatDb();
});
afterEach(async () => {
  closeChatDb();
  if (saved === undefined) delete process.env.STOCKBOT_DATA_DIR;
  else process.env.STOCKBOT_DATA_DIR = saved;
  await rm(directory, { recursive: true, force: true });
});

function get(query = ""): Response {
  return GET(new Request(`http://localhost/api/chat${query}`));
}
function remove(query = ""): Response {
  return DELETE(new Request(`http://localhost/api/chat${query}`, { method: "DELETE" }));
}

describe("chat route", () => {
  test("lists stored chats newest first without their messages", async () => {
    const chatId = createChat("NVDA filings");
    appendMessage(chatId, "user", "What did NVDA file?");
    const body = (await get().json()) as { chats: { chatId: string; title: string; updatedAt: string }[] };
    expect(body.chats).toEqual([{ chatId, title: "NVDA filings", updatedAt: expect.any(String) }]);
  });

  test("reads one chat with its stored messages, and answers 404 for an unknown id", async () => {
    const chatId = createChat("NVDA filings");
    appendMessage(chatId, "user", "What did NVDA file?");
    appendMessage(chatId, "assistant", "An 8-K on 2 September.");
    expect(await get(`?id=${chatId}`).json()).toMatchObject({
      chatId,
      title: "NVDA filings",
      messages: [{ role: "user", content: "What did NVDA file?" }, { role: "assistant", content: "An 8-K on 2 September." }],
    });
    const missing = get("?id=00000000-0000-4000-8000-000000000000");
    expect(missing.status).toBe(404);
    expect(await missing.json()).toEqual({ error: "chat not found" });
  });

  test("deletes one chat, rejects an unknown one, and requires an id", async () => {
    const chatId = createChat("delete me");
    expect(await remove(`?id=${chatId}`).json()).toEqual({ deleted: chatId });
    expect((await get().json() as { chats: unknown[] }).chats).toEqual([]);
    expect(remove(`?id=${chatId}`).status).toBe(404);
    const noId = remove();
    expect(noId.status).toBe(400);
    expect(await noId.json()).toEqual({ error: "id required" });
  });
});
