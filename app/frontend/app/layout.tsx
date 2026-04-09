import type { Metadata } from "next";
import { Geist } from "next/font/google";
import "./globals.css";
import Nav from "@/components/Nav";

const geist = Geist({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Mouna — ASL Sign Language Recognition",
  description: "Real-time American Sign Language recognition powered by MediaPipe and AI",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${geist.className} h-full`}>
      <body className="min-h-full flex bg-gray-100 text-gray-900">
        <Nav />
        <main className="flex-1 overflow-y-auto">{children}</main>
      </body>
    </html>
  );
}
