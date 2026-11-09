import { deleteChat, listChats, readChat } from "@/lib/chat";

// Conversation history of this local installation. The user reads and deletes it from the chat panel.
// One query parameter, `id`, selects one chat. No chat is ever created here: the agent route creates it.
export function GET(req: Request): Response {
  const chatId = new URL(req.url).searchParams.get("id");
  if (chatId === null) return Response.json({ chats: listChats() }, { headers: { "Cache-Control": "no-store" } });
  const chat = readChat(chatId);
  if (!chat) return Response.json({ error: "chat not found" }, { status: 404 });
  return Response.json(chat, { headers: { "Cache-Control": "no-store" } });
}

// Deletion is the user's own action; a missing chat answers 404 instead of pretending success.
export function DELETE(req: Request): Response {
  const chatId = new URL(req.url).searchParams.get("id");
  if (chatId === null) return Response.json({ error: "id required" }, { status: 400 });
  if (!deleteChat(chatId)) return Response.json({ error: "chat not found" }, { status: 404 });
  return Response.json({ deleted: chatId });
}
