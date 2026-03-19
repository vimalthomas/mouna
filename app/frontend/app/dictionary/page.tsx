"use client";

import { useState, useEffect, useMemo } from "react";
import SignCard from "@/components/SignCard";
import { Sign, SignsData } from "@/lib/types";

export default function DictionaryPage() {
  const [signs, setSigns] = useState<Sign[]>([]);
  const [categories, setCategories] = useState<string[]>([]);
  const [search, setSearch] = useState("");
  const [category, setCategory] = useState("all");
  const [difficulty, setDifficulty] = useState("all");

  useEffect(() => {
    fetch("http://localhost:8000/signs")
      .then((r) => r.json())
      .then((data: SignsData) => {
        setSigns(data.signs);
        setCategories(data.categories);
      })
      .catch(() => {});
  }, []);

  const filtered = useMemo(
    () =>
      signs.filter((s) => {
        if (search && !s.word.includes(search.toLowerCase())) return false;
        if (category !== "all" && s.category !== category) return false;
        if (difficulty !== "all" && s.difficulty !== difficulty) return false;
        return true;
      }),
    [signs, search, category, difficulty]
  );

  return (
    <div className="p-6 max-w-6xl mx-auto w-full">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-white mb-1">Sign Dictionary</h1>
        <p className="text-zinc-500 text-sm">{signs.length} signs across {categories.length} categories</p>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap gap-3 mb-6">
        <input
          type="text"
          placeholder="Search signs…"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="bg-zinc-900 border border-zinc-700 rounded-lg px-4 py-2 text-sm text-white placeholder-zinc-500 focus:outline-none focus:border-violet-500 w-48"
        />
        <select
          value={category}
          onChange={(e) => setCategory(e.target.value)}
          className="bg-zinc-900 border border-zinc-700 rounded-lg px-4 py-2 text-sm text-white focus:outline-none focus:border-violet-500"
        >
          <option value="all">All categories</option>
          {categories.map((c) => (
            <option key={c} value={c} className="capitalize">{c}</option>
          ))}
        </select>
        <select
          value={difficulty}
          onChange={(e) => setDifficulty(e.target.value)}
          className="bg-zinc-900 border border-zinc-700 rounded-lg px-4 py-2 text-sm text-white focus:outline-none focus:border-violet-500"
        >
          <option value="all">All difficulties</option>
          <option value="easy">Easy</option>
          <option value="medium">Medium</option>
          <option value="hard">Hard</option>
        </select>
        {(search || category !== "all" || difficulty !== "all") && (
          <button
            onClick={() => { setSearch(""); setCategory("all"); setDifficulty("all"); }}
            className="text-sm text-zinc-400 hover:text-white px-3 py-2"
          >
            Clear filters
          </button>
        )}
      </div>

      {/* Results count */}
      <p className="text-xs text-zinc-600 mb-4">
        {filtered.length} sign{filtered.length !== 1 ? "s" : ""}
      </p>

      {/* Grid */}
      {filtered.length === 0 ? (
        <div className="text-center py-20 text-zinc-600">No signs match your filters.</div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
          {filtered.map((sign) => (
            <SignCard key={sign.word} sign={sign} />
          ))}
        </div>
      )}
    </div>
  );
}
