import { SignsData } from "@/lib/types";

export async function fetchSigns(): Promise<SignsData> {
  try {
    const res = await fetch("http://localhost:8000/signs");
    if (res.ok) return res.json();
  } catch {
    // backend not running — fall back to local data
  }
  const res = await fetch("/signs.json");
  const data = await res.json();
  // derive categories from signs if not present
  if (!data.categories) {
    data.categories = [...new Set<string>(data.signs.map((s: { category: string }) => s.category))].sort();
  }
  return data;
}
