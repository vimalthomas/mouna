"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/", label: "Dashboard" },
  { href: "/practice", label: "Practice" },
  { href: "/dictionary", label: "Dictionary" },
  { href: "/coach", label: "AI Coach" },
];

export default function Nav() {
  const pathname = usePathname();
  return (
    <nav className="w-44 shrink-0 bg-[#1e1b2e] min-h-screen flex flex-col py-6 px-4 gap-6">
      <Link href="/" className="text-violet-400 font-bold text-xl tracking-wide">
        MOUNA
      </Link>
      <div className="flex flex-col gap-1">
        {links.map(({ href, label }) => {
          const active = pathname === href;
          return (
            <Link
              key={href}
              href={href}
              className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                active
                  ? "bg-zinc-600/40 text-white"
                  : "text-zinc-300 hover:text-white hover:bg-zinc-700/30"
              }`}
            >
              {label}
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
