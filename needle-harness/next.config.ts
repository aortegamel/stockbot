import { networkInterfaces } from "node:os";
import type { NextConfig } from "next";

const allowedDevOrigins = Object.values(networkInterfaces())
  .flat()
  .filter((n) => n?.family === "IPv4" && !n.internal)
  .map((n) => n!.address);

const nextConfig: NextConfig = {
  allowedDevOrigins,
  // Dev + prod: /api/* proxies to the FastAPI real-data backend so the
  // harness never needs NEXT_PUBLIC_API_URL set locally.
  async rewrites() {
    return [{ source: "/api/:path*", destination: "http://127.0.0.1:8000/api/:path*" }];
  },
  turbopack: {
    root: __dirname,
  },
};

export default nextConfig;
