"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/", label: "Detect" },
  { href: "/dictionary", label: "Dictionary" },
  { href: "/practice", label: "Practice" },
];

export default function Nav() {
  const pathname = usePathname();
  return (
    <nav className="flex items-center justify-between px-6 py-4 border-b border-zinc-800 bg-zinc-950">
      <Link href="/" className="flex items-center gap-2">
        <span className="text-2xl">🤟</span>
        <span className="text-xl font-bold tracking-tight text-white">Mouna</span>
        <span className="text-xs text-zinc-500 mt-0.5">ASL</span>
      </Link>
      <div className="flex items-center gap-1">
        {links.map(({ href, label }) => (
          <Link
            key={href}
            href={href}
            className={`px-4 py-1.5 rounded-full text-sm font-medium transition-colors ${
              pathname === href
                ? "bg-violet-600 text-white"
                : "text-zinc-400 hover:text-white hover:bg-zinc-800"
            }`}
          >
            {label}
          </Link>
        ))}
      </div>
    </nav>
  );
}
