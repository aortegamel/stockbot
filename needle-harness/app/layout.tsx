import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Needle Terminal",
  description: "Dark enterprise trading terminal (mock data)",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
