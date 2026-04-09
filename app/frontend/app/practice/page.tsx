"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import { useRouter } from "next/navigation";
import Camera from "@/components/Camera";
import { Sign, InferenceResult } from "@/lib/types";
import { fetchSigns } from "@/lib/fetchSigns";

type Mode = "words" | "letters" | "sentences";

const LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("");

interface VideoInfo {
  url: string;
  start: number | null;
  end: number | null;
}

export default function PracticePage() {
  const router = useRouter();
  const [, setSigns] = useState<Sign[]>([]);
  const [, setCategories] = useState<string[]>([]);
  const [mode, setMode] = useState<Mode>("words");
  const [target, setTarget] = useState<Sign | null>(null);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [index, setIndex] = useState(0);
  const [sessionSigns, setSessionSigns] = useState<Sign[]>([]);
  const [videoInfo, setVideoInfo] = useState<VideoInfo | null | undefined>(undefined);
  const videoRef = useRef<HTMLVideoElement>(null);

  // Letters mode state
  const [letterIndex, setLetterIndex] = useState(0);

  useEffect(() => {
    fetchSigns()
      .then((data) => {
        setSigns(data.signs);
        setCategories(data.categories);
        const shuffled = [...data.signs].sort(() => Math.random() - 0.5);
        const session = shuffled.slice(0, 11);
        setSessionSigns(session);
        if (session.length > 0) {
          setTarget(session[0]);
          setIndex(0);
        }
      })
      .catch(() => {});
  }, []);

  // Load reference video from local public folder
  useEffect(() => {
    if (!target || mode === "letters") return;
    if (target.video) {
      setVideoInfo({ url: target.video, start: null, end: null });
    } else {
      setVideoInfo(null);
    }
  }, [target, mode]);

  const handleResult = useCallback((r: InferenceResult) => {
    setResult(r);
  }, []);

  const retrySign = useCallback(() => {
    setResult(null);
  }, []);

  function handleVideoLoaded() {
    if (!videoRef.current || !videoInfo) return;
    videoRef.current.currentTime = videoInfo.start ?? 0;
  }

  function handleTimeUpdate() {
    if (!videoRef.current || !videoInfo || videoInfo.end == null) return;
    if (videoRef.current.currentTime >= videoInfo.end) {
      videoRef.current.pause();
      videoRef.current.currentTime = videoInfo.start ?? 0;
    }
  }

  function playReference() {
    videoRef.current?.play().catch(() => {});
  }

  // Compute progress based on mode
  const progressPct =
    mode === "letters"
      ? ((letterIndex + 1) / LETTERS.length) * 100
      : sessionSigns.length > 0
      ? ((index + 1) / sessionSigns.length) * 100
      : 0;

  const progressLabel =
    mode === "letters"
      ? `Progress: Sign ${letterIndex + 1} of ${LETTERS.length}`
      : `Progress: Sign ${index + 1} of ${sessionSigns.length}`;

  const category = sessionSigns.length > 0 && target ? target.category : "";

  // Current display word
  const displayWord =
    mode === "letters" ? LETTERS[letterIndex] : target?.word?.toUpperCase() ?? "";

  return (
    <div className="p-8 max-w-5xl mx-auto w-full">
      <h1 className="text-2xl font-bold text-gray-900 mb-4">Practice Mode</h1>

      {/* Mode Tabs */}
      <div className="flex items-center gap-2 mb-6 bg-gray-200/60 rounded-full p-1 w-fit">
        {(["words", "letters", "sentences"] as Mode[]).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className={`px-6 py-2 rounded-full text-sm font-medium transition-colors capitalize ${
              mode === m
                ? "bg-violet-600 text-white shadow-sm"
                : "text-gray-600 hover:text-gray-900"
            }`}
          >
            {m}
          </button>
        ))}
      </div>

      {/* Session Info */}
      <div className="mb-2">
        <p className="text-violet-600 text-sm font-medium capitalize">
          {mode === "letters"
            ? "Alphabet Practice"
            : category
            ? `${category} Vocabulary`
            : "Practice Session"}{" "}
          &middot; Real-time sign recognition
        </p>
      </div>

      {/* Progress Bar */}
      <div className="mb-1">
        <div className="h-2.5 bg-gray-200 rounded-full overflow-hidden">
          <div
            className="h-full rounded-full transition-all"
            style={{
              width: `${progressPct}%`,
              background: "linear-gradient(90deg, #4f46e5, #a78bfa)",
            }}
          />
        </div>
      </div>
      <p className="text-gray-400 text-xs mb-6">{progressLabel}</p>

      {/* Letter Grid (Letters mode only) */}
      {mode === "letters" && (
        <div className="flex flex-wrap gap-2 justify-center mb-6">
          {LETTERS.map((letter, i) => {
            let style = "border border-gray-300 bg-white text-gray-700";
            if (i < letterIndex) {
              style = "bg-violet-600 text-white";
            } else if (i === letterIndex) {
              style = "bg-violet-400 text-white";
            }
            return (
              <button
                key={letter}
                onClick={() => setLetterIndex(i)}
                className={`w-10 h-10 rounded-lg text-sm font-bold flex items-center justify-center transition-colors ${style}`}
              >
                {letter}
              </button>
            );
          })}
        </div>
      )}

      {/* Target Word / Letter */}
      {displayWord && mode !== "letters" && (
        <h2 className="text-4xl font-bold text-violet-600 text-center mb-6 uppercase tracking-wide">
          {displayWord}
        </h2>
      )}

      {/* Sentences placeholder */}
      {mode === "sentences" && (
        <div className="bg-violet-50 border border-violet-200 rounded-2xl px-6 py-8 text-center mb-6">
          <p className="text-violet-600 font-semibold text-sm">Sentence practice coming soon</p>
          <p className="text-gray-400 text-xs mt-1">Switch to Words or Letters to continue practicing</p>
        </div>
      )}

      {/* Side-by-side: Camera + Reference Video */}
      {mode !== "sentences" && (<div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-4">
        {/* Camera */}
        <div>
          <div className="flex items-center gap-2 mb-2">
            <span className="w-2.5 h-2.5 rounded-full bg-red-500" />
            <span className="text-sm font-medium text-gray-700">Camera</span>
          </div>
          <Camera onResult={handleResult} />
          <div className="flex gap-3 mt-3">
            <button
              onClick={retrySign}
              className="px-6 py-2.5 rounded-full text-sm font-medium border border-gray-300 text-gray-700 hover:bg-gray-50 transition-colors"
            >
              Retry
            </button>
          </div>
        </div>

        {/* Reference Video */}
        <div>
          <p className="text-sm font-medium text-gray-700 mb-2">Reference Video</p>
          <div className="bg-gray-200 rounded-xl aspect-video flex items-center justify-center overflow-hidden">
            {videoInfo === undefined && mode !== "letters" && (
              <div className="flex items-center gap-2 text-gray-400 text-sm">
                <div className="w-4 h-4 border-2 border-gray-400 border-t-transparent rounded-full animate-spin" />
                Loading...
              </div>
            )}
            {(videoInfo === null || mode === "letters") && (
              <div className="flex items-center gap-2 text-gray-400 text-sm">
                <svg className="w-6 h-6" fill="currentColor" viewBox="0 0 24 24">
                  <path d="M8 5v14l11-7z" />
                </svg>
                <span>Reference Sign</span>
              </div>
            )}
            {videoInfo && mode !== "letters" && (
              <video
                ref={videoRef}
                key={videoInfo.url}
                src={videoInfo.url}
                muted
                loop
                className="w-full h-full object-cover"
                onLoadedMetadata={handleVideoLoaded}
                onTimeUpdate={handleTimeUpdate}
              />
            )}
          </div>
          <div className="flex gap-3 mt-3">
            <button
              onClick={playReference}
              className="bg-violet-600 hover:bg-violet-500 text-white text-sm font-medium px-6 py-2.5 rounded-full transition-colors"
            >
              Play
            </button>
            <button
              onClick={() => {
                if (videoRef.current && videoInfo) {
                  videoRef.current.currentTime = videoInfo.start ?? 0;
                }
              }}
              className="px-6 py-2.5 rounded-full text-sm font-medium border border-gray-300 text-gray-700 hover:bg-gray-50 transition-colors"
            >
              Retry
            </button>
          </div>
        </div>
      </div>)}

      {/* Result feedback */}
      {result && (
        <div className={`rounded-2xl border px-6 py-4 mt-4 flex items-center justify-between ${
          result.uncertain ? "bg-amber-50 border-amber-200" :
          result.prediction === target?.word ? "bg-green-50 border-green-200" : "bg-red-50 border-red-200"
        }`}>
          <div>
            <p className="text-sm font-semibold text-gray-700">
              {result.uncertain ? "Unclear sign — try again" :
               result.prediction === target?.word ? `✓ Correct! You signed "${result.prediction}"` :
               `Looks like "${result.prediction}" — expected "${target?.word}"`}
            </p>
            <p className="text-xs text-gray-400 mt-0.5">Confidence: {Math.round(result.confidence * 100)}%</p>
          </div>
          <button onClick={() => setResult(null)} className="text-xs text-gray-400 hover:text-gray-600 ml-4">
            Dismiss
          </button>
        </div>
      )}

      {/* Practice with AI */}
      <div className="flex justify-center gap-4 mt-6">
        <button
          onClick={() => router.push("/coach")}
          className="bg-violet-600 hover:bg-violet-500 text-white text-sm font-semibold px-8 py-3 rounded-full transition-colors"
        >
          Practice with AI
        </button>
      </div>
    </div>
  );
}
