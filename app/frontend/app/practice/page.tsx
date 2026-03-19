"use client";

import { useState, useEffect, useCallback } from "react";
import Camera from "@/components/Camera";
import SignCard from "@/components/SignCard";
import { Sign, SignsData, InferenceResult } from "@/lib/types";

type PracticeResult = "correct" | "wrong" | "uncertain" | null;

export default function PracticePage() {
  const [signs, setSigns] = useState<Sign[]>([]);
  const [target, setTarget] = useState<Sign | null>(null);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [outcome, setOutcome] = useState<PracticeResult>(null);
  const [score, setScore] = useState({ correct: 0, total: 0 });
  const [category, setCategory] = useState("all");
  const [categories, setCategories] = useState<string[]>([]);

  useEffect(() => {
    fetch("http://localhost:8000/signs")
      .then((r) => r.json())
      .then((data: SignsData) => {
        setSigns(data.signs);
        setCategories(data.categories);
      })
      .catch(() => {});
  }, []);

  const pickTarget = useCallback(
    (pool: Sign[]) => {
      const filtered = category === "all" ? pool : pool.filter((s) => s.category === category);
      if (!filtered.length) return;
      const pick = filtered[Math.floor(Math.random() * filtered.length)];
      setTarget(pick);
      setResult(null);
      setOutcome(null);
    },
    [category]
  );

  useEffect(() => {
    if (signs.length) pickTarget(signs);
  }, [signs, pickTarget]);

  const handleResult = useCallback(
    (r: InferenceResult) => {
      setResult(r);
      if (r.uncertain) {
        setOutcome("uncertain");
        setScore((s) => ({ ...s, total: s.total + 1 }));
      } else if (r.prediction === target?.word) {
        setOutcome("correct");
        setScore((s) => ({ correct: s.correct + 1, total: s.total + 1 }));
      } else {
        setOutcome("wrong");
        setScore((s) => ({ ...s, total: s.total + 1 }));
      }
    },
    [target]
  );

  const outcomeConfig = {
    correct: { bg: "bg-emerald-950 border-emerald-700", icon: "✓", text: "text-emerald-400", label: "Correct!" },
    wrong: { bg: "bg-red-950 border-red-800", icon: "✗", text: "text-red-400", label: `Wrong — it was "${result?.prediction}"` },
    uncertain: { bg: "bg-zinc-900 border-zinc-700", icon: "?", text: "text-zinc-400", label: "Uncertain — try again" },
  };

  return (
    <div className="flex flex-col lg:flex-row gap-6 p-6 max-w-6xl mx-auto w-full flex-1">
      {/* Left: target + camera */}
      <div className="flex-1 min-w-0 flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <h1 className="text-xl font-semibold text-white">Practice Mode</h1>
          <div className="flex items-center gap-3">
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              className="bg-zinc-900 border border-zinc-700 rounded-lg px-3 py-1.5 text-sm text-white focus:outline-none focus:border-violet-500"
            >
              <option value="all">All categories</option>
              {categories.map((c) => (
                <option key={c} value={c} className="capitalize">{c}</option>
              ))}
            </select>
            <span className="text-sm text-zinc-400 font-mono">
              {score.correct}/{score.total}
            </span>
          </div>
        </div>

        {/* Score bar */}
        {score.total > 0 && (
          <div className="h-1.5 bg-zinc-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-emerald-500 rounded-full transition-all"
              style={{ width: `${(score.correct / score.total) * 100}%` }}
            />
          </div>
        )}

        {/* Target sign */}
        {target && (
          <div className="rounded-2xl border border-violet-700 bg-violet-950/40 p-4">
            <p className="text-xs text-violet-400 uppercase tracking-widest mb-1">Sign this word</p>
            <p className="text-3xl font-bold text-white capitalize mb-2">{target.word}</p>
            <p className="text-sm text-zinc-400">{target.description}</p>
          </div>
        )}

        <Camera onResult={handleResult} />
      </div>

      {/* Right: feedback */}
      <div className="w-full lg:w-72 shrink-0 flex flex-col gap-4">
        <h2 className="text-xl font-semibold text-white">Feedback</h2>

        {outcome && result && (
          <div className={`rounded-2xl border p-5 ${outcomeConfig[outcome].bg}`}>
            <div className="flex items-center gap-3 mb-3">
              <span className={`text-3xl font-bold ${outcomeConfig[outcome].text}`}>
                {outcomeConfig[outcome].icon}
              </span>
              <span className={`font-semibold ${outcomeConfig[outcome].text}`}>
                {outcomeConfig[outcome].label}
              </span>
            </div>
            <div className="text-sm text-zinc-400">
              {result.top3.map((p, i) => (
                <div key={i} className="flex justify-between py-0.5">
                  <span className="capitalize">{p.label}</span>
                  <span>{Math.round(p.confidence * 100)}%</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {!outcome && (
          <div className="rounded-2xl border border-zinc-800 bg-zinc-900 p-5 flex flex-col items-center justify-center min-h-40 gap-2">
            <span className="text-3xl">🎯</span>
            <p className="text-zinc-500 text-sm text-center">Record the target sign to get feedback</p>
          </div>
        )}

        <button
          onClick={() => pickTarget(signs)}
          className="w-full py-3 rounded-xl font-semibold text-sm bg-zinc-800 hover:bg-zinc-700 text-white transition-colors"
        >
          Next Sign →
        </button>

        {target && (
          <div className="mt-2">
            <p className="text-xs text-zinc-600 mb-2 uppercase tracking-widest">Reference</p>
            <SignCard sign={target} highlight />
          </div>
        )}
      </div>
    </div>
  );
}
