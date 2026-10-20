import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { lanTokenMiddleware } from "./lan-token";

const api = `http://127.0.0.1:${process.env.STOCKBOT_API_PORT ?? "3001"}`;
const market = `http://127.0.0.1:${process.env.STOCKBOT_MARKET_PORT ?? "8000"}`;
const to = (target: string) => ({ target, changeOrigin: true });
const proxy = {
  "/api/agent": to(api),
  "/api/watchlist": to(api),
  "/api/research": to(api),
  "/api/health": to(api),
  "/api/candles": to(market),
  "/api/quote": to(market),
  "/api/search": to(market),
  "/api/universe": to(market),
  "/api/identity": to(market),
  "/api/news": to(market),
  "/api/filings": to(market),
  "/api/viewer-context": to(market),
  "/api/theses": to(market),
  "/api/portfolio": to(market),
  "/api/sparklines": to(market),
  "/api/markets": to(market),
  "/api/scans": to(market),
  "/api/briefing": to(market),
  "/api/feed": to(market),
  "/api/positioning": to(market),
  "/api/workspaces": to(market),
  "/api/widget-data": to(market),
};
// Host comes from the launcher --host flag. The gate runs before the proxy in dev and preview.
const gate = lanTokenMiddleware();

export default defineConfig({
  plugins: [
    react(),
    tailwindcss(),
    {
      name: "stockbot-lan-token",
      configureServer(server) {
        server.middlewares.use(gate);
      },
      configurePreviewServer(server) {
        server.middlewares.use(gate);
      },
    },
  ],
  server: { proxy },
  preview: { proxy },
});
