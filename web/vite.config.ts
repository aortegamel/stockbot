import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const api = `http://127.0.0.1:${process.env.STOCKBOT_API_PORT ?? "3001"}`;
const market = `http://127.0.0.1:${process.env.STOCKBOT_MARKET_PORT ?? "8000"}`;
const proxy = (target: string) => ({ target, changeOrigin: true });

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    proxy: {
      "/api/agent": proxy(api),
      "/api/research": proxy(api),
      "/api/health": proxy(api),
      "/api/candles": proxy(market),
      "/api/quote": proxy(market),
      "/api/search": proxy(market),
      "/api/universe": proxy(market),
      "/api/identity": proxy(market),
      "/api/news": proxy(market),
      "/api/filings": proxy(market),
      "/api/viewer-context": proxy(market),
    },
  },
});
