"use client";

import { useState, useEffect } from "react";
import Camera from "@/components/Camera";
import Results from "@/components/Results";
import { InferenceResult, Sign, SignsData } from "@/lib/types";

export default function DetectPage() {
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [signs, setSigns] = useState<Sign[]>([]);

  useEffect(() => {
    fetch("http://localhost:8000/signs")
      .then((r) => r.json())
      .then((data: SignsData) => setSigns(data.signs))
      .catch(() => {});
  }, []);

  return (
    <div className="flex flex-col lg:flex-row gap-6 p-6 max-w-6xl mx-auto w-full flex-1">
      <div className="flex-1 min-w-0">
        <h1 className="text-xl font-semibold text-white mb-4">Live Detection</h1>
        <Camera onResult={setResult} />
        <p className="mt-3 text-xs text-zinc-600 text-center">
          Press Record, perform a sign within 3 seconds, and Mouna will identify it.
        </p>
      </div>

      <div className="w-full lg:w-80 shrink-0">
        <h2 className="text-xl font-semibold text-white mb-4">Result</h2>
        <div className="rounded-2xl border border-zinc-800 bg-zinc-900 p-5 min-h-64">
          <Results result={result} signs={signs} idle={!result} />
        </div>
        {result && !result.uncertain && (
          <div className="mt-3 text-xs text-zinc-600 text-center">
            {Math.round(result.confidence * 100)}% confident
            {result.confidence < 0.5 && " — try signing more clearly"}
          </div>
        )}
      </div>
    </div>
  );
}
