"use client";

import { useState, useEffect, useMemo, useRef } from "react";
import { Sign } from "@/lib/types";
import { fetchSigns } from "@/lib/fetchSigns";

function VideoCard({ sign }: { sign: Sign }) {
  const [playing, setPlaying] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);

  function handlePlay() {
    if (!sign.video) return;
    setPlaying(true);
    videoRef.current?.play().catch(() => {});
  }

  return (
    <div className="flex flex-col">
      <div
        onClick={handlePlay}
        className="bg-white border border-gray-200 rounded-xl aspect-square flex items-center justify-center cursor-pointer hover:border-gray-300 transition-colors overflow-hidden relative"
      >
        {sign.video ? (
          <video
            ref={videoRef}
            src={sign.video}
            muted
            loop
            className="w-full h-full object-cover"
            onEnded={() => setPlaying(false)}
          />
        ) : (
          <svg className="w-10 h-10 text-gray-300" fill="currentColor" viewBox="0 0 24 24">
            <path d="M8 5v14l11-7z" />
          </svg>
        )}
        {!playing && sign.video && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/10">
            <svg className="w-10 h-10 text-white drop-shadow" fill="currentColor" viewBox="0 0 24 24">
              <path d="M8 5v14l11-7z" />
            </svg>
          </div>
        )}
      </div>
      <p className="text-sm font-semibold text-gray-900 mt-2 text-center capitalize">{sign.word}</p>
      <p className="text-xs text-gray-400 text-center">{sign.description}</p>
    </div>
  );
}

export default function DictionaryPage() {
  const [signs, setSigns] = useState<Sign[]>([]);
  const [categories, setCategories] = useState<string[]>([]);
  const [search, setSearch] = useState("");
  const [category, setCategory] = useState("all");

  useEffect(() => {
    fetchSigns()
      .then((data) => {
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
        return true;
      }),
    [signs, search, category]
  );

  return (
    <div className="p-8 max-w-5xl mx-auto w-full">
      <h1 className="text-2xl font-bold text-gray-900 mb-6">Dictionary</h1>

      {/* Search */}
      <input
        type="text"
        placeholder="Search Signs.."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        className="w-full bg-white border border-gray-200 rounded-2xl px-5 py-3.5 text-sm text-gray-900 placeholder-gray-400 focus:outline-none focus:border-violet-400 mb-6"
      />

      {/* Category Pills */}
      <div className="flex items-center gap-2 mb-8 overflow-x-auto bg-gray-200/60 rounded-full p-1 w-fit">
        <button
          onClick={() => setCategory("all")}
          className={`px-5 py-2 rounded-full text-sm font-medium transition-colors whitespace-nowrap ${
            category === "all"
              ? "bg-white text-violet-600 shadow-sm"
              : "text-gray-600 hover:text-gray-900"
          }`}
        >
          All
        </button>
        {categories.map((c) => (
          <button
            key={c}
            onClick={() => setCategory(c)}
            className={`px-5 py-2 rounded-full text-sm font-medium transition-colors whitespace-nowrap capitalize ${
              category === c
                ? "bg-violet-600 text-white shadow-sm"
                : "text-gray-600 hover:text-gray-900"
            }`}
          >
            {c}
          </button>
        ))}
      </div>

      {/* Grid */}
      {filtered.length === 0 ? (
        <div className="text-center py-20 text-gray-400">No signs match your search.</div>
      ) : (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-6">
          {filtered.map((sign) => (
            <VideoCard key={sign.word} sign={sign} />
          ))}
        </div>
      )}
    </div>
  );
}
