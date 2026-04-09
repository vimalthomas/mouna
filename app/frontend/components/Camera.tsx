"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { InferenceResult } from "@/lib/types";

const RECORD_SECS = 3;
const MOCK_SIGNS = ["book", "drink", "computer"];

type Status = "loading" | "ready" | "recording" | "processing" | "error";

interface Props {
  onResult?: (result: InferenceResult) => void;
}

export default function Camera({ onResult }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [status, setStatus] = useState<Status>("loading");
  const [countdown, setCountdown] = useState(0);
  const [error, setError] = useState("");

  const rafRef = useRef<number>(0);

  // Start camera on mount
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        if (cancelled) { stream.getTracks().forEach((t) => t.stop()); return; }
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }
        setStatus("ready");
      } catch (err) {
        if (!cancelled) {
          setError("Camera access denied. Please allow camera permission and reload.");
          setStatus("error");
        }
      }
    })();
    return () => {
      cancelled = true;
      cancelAnimationFrame(rafRef.current);
      if (videoRef.current?.srcObject) {
        (videoRef.current.srcObject as MediaStream).getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  const startRecording = useCallback(() => {
    if (status !== "ready") return;
    setStatus("recording");
    setCountdown(RECORD_SECS);

    const tick = setInterval(() => {
      setCountdown((c) => {
        if (c <= 1) clearInterval(tick);
        return c - 1;
      });
    }, 1000);

    setTimeout(() => {
      clearInterval(tick);
      setStatus("processing");

      // Mock inference after short delay
      setTimeout(() => {
        const prediction = MOCK_SIGNS[Math.floor(Math.random() * MOCK_SIGNS.length)];
        const confidence = 0.55 + Math.random() * 0.35;
        const others = MOCK_SIGNS.filter((s) => s !== prediction);
        const mockResult: InferenceResult = {
          prediction,
          confidence,
          uncertain: confidence < 0.6,
          top3: [
            { label: prediction, confidence },
            { label: others[0], confidence: confidence * 0.45 },
            { label: others[1], confidence: confidence * 0.18 },
          ],
        };
        onResult?.(mockResult);
        setStatus("ready");
      }, 800);
    }, RECORD_SECS * 1000);
  }, [status, onResult]);

  return (
    <div className="flex flex-col gap-4">
      <div className="relative rounded-2xl overflow-hidden bg-zinc-900 aspect-video">
        <video
          ref={videoRef}
          className="w-full h-full object-cover scale-x-[-1]"
          playsInline
          muted
        />

        {status === "loading" && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-zinc-950/90 gap-3">
            <div className="w-8 h-8 border-2 border-violet-500 border-t-transparent rounded-full animate-spin" />
            <p className="text-zinc-400 text-sm">Starting camera…</p>
          </div>
        )}

        {status === "error" && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-zinc-950/90 gap-2 px-6">
            <span className="text-3xl">⚠️</span>
            <p className="text-red-400 text-sm text-center">{error}</p>
          </div>
        )}

        {status === "recording" && (
          <div className="absolute top-3 left-3 flex items-center gap-2 bg-red-600/90 px-3 py-1 rounded-full">
            <span className="w-2 h-2 rounded-full bg-white animate-pulse" />
            <span className="text-white text-xs font-semibold">RECORDING {countdown}s</span>
          </div>
        )}

        {status === "processing" && (
          <div className="absolute inset-0 flex items-center justify-center bg-zinc-950/60">
            <div className="flex items-center gap-2 bg-zinc-900 px-4 py-2 rounded-full">
              <div className="w-4 h-4 border-2 border-violet-400 border-t-transparent rounded-full animate-spin" />
              <span className="text-zinc-300 text-sm">Analyzing…</span>
            </div>
          </div>
        )}
      </div>

      <button
        onClick={startRecording}
        disabled={status !== "ready"}
        className={`w-full py-3 rounded-xl font-semibold text-sm transition-all ${
          status === "ready"
            ? "bg-violet-600 hover:bg-violet-500 text-white cursor-pointer"
            : "bg-zinc-800 text-zinc-500 cursor-not-allowed"
        }`}
      >
        {status === "recording"
          ? `Recording… ${countdown}s`
          : status === "processing"
          ? "Analyzing…"
          : status === "loading"
          ? "Loading…"
          : "Record Sign (3s)"}
      </button>
    </div>
  );
}
