"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import Camera from "@/components/Camera";
import { Sign, InferenceResult } from "@/lib/types";
import { fetchSigns } from "@/lib/fetchSigns";

interface VideoInfo {
  url: string;
  start: number | null;
  end: number | null;
}

export default function CoachPage() {
  const [signs, setSigns] = useState<Sign[]>([]);
  const [target, setTarget] = useState<Sign | null>(null);
  const [feedback, setFeedback] = useState<string | null>(null);
  const [started, setStarted] = useState(false);
  const [videoInfo, setVideoInfo] = useState<VideoInfo | null | undefined>(undefined);
  const videoRef = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    fetchSigns()
      .then((data) => {
        setSigns(data.signs);
        // Pick first available sign
        if (data.signs.length > 0) setTarget(data.signs[0]);
      })
      .catch(() => {});
  }, []);

  // Load reference video from local public folder
  useEffect(() => {
    if (!target) return;
    if (target.video) {
      setVideoInfo({ url: target.video, start: null, end: null });
    } else {
      setVideoInfo(null);
    }
  }, [target]);

  const handleResult = useCallback(
    (r: InferenceResult) => {
      if (!target) return;
      if (r.uncertain) {
        setFeedback("The sign wasn't clear enough. Try slowing down and making each motion more distinct.");
      } else if (r.prediction === target.word) {
        setFeedback(`Great job! You signed "${target.word}" correctly with ${Math.round(r.confidence * 100)}% confidence.`);
      } else {
        setFeedback(`That looked like "${r.prediction}" instead of "${target.word}". Focus on the hand shape and movement direction.`);
      }
    },
    [target]
  );

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

  function handleBegin() {
    setStarted(true);
    videoRef.current?.play().catch(() => {});
  }

  function pickNextSign() {
    if (signs.length === 0) return;
    const others = signs.filter((s) => s.word !== target?.word);
    const pool = others.length > 0 ? others : signs;
    const pick = pool[Math.floor(Math.random() * pool.length)];
    setTarget(pick);
    setFeedback(null);
    setStarted(false); // reset so "Click to Begin" shows again for new sign
  }

  return (
    <div className="p-8 max-w-5xl mx-auto w-full">
      <div className="mb-6">
        <h1 className="text-2xl font-bold text-gray-900">AI Coach</h1>
        <p className="text-gray-500 text-sm mt-0.5">
          Sign along with the expert — match in real time
        </p>
      </div>

      {/* Target Word + Badge */}
      {target && (
        <div className="mb-6">
          <h2 className="text-4xl font-bold text-violet-600 uppercase tracking-wide">
            {target.word}
          </h2>
          <span className="inline-block mt-2 bg-violet-600 text-white text-xs font-semibold px-3 py-1 rounded-full">
            weak area
          </span>
        </div>
      )}

      {/* Click to Begin */}
      {!started && (
        <div className="flex justify-center mb-4">
          <button
            onClick={handleBegin}
            className="bg-violet-600 hover:bg-violet-500 text-white text-sm font-semibold px-6 py-2.5 rounded-full transition-colors"
          >
            Click to Begin
          </button>
        </div>
      )}

      {/* Side-by-side: You + Expert Sign */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
        {/* Camera (You) */}
        <div>
          <div className="flex items-center gap-2 mb-2">
            <span className="w-2.5 h-2.5 rounded-full bg-red-500" />
            <span className="text-sm font-medium text-gray-700">You</span>
          </div>
          <Camera onResult={handleResult} />
        </div>

        {/* Expert Sign */}
        <div>
          <p className="text-sm font-medium text-gray-700 mb-2">Expert Sign</p>
          <div className="bg-gray-200 rounded-xl aspect-video flex items-center justify-center overflow-hidden relative">
            {videoInfo === undefined && (
              <div className="flex items-center gap-2 text-gray-400 text-sm">
                <div className="w-4 h-4 border-2 border-gray-400 border-t-transparent rounded-full animate-spin" />
                Loading...
              </div>
            )}
            {videoInfo === null && (
              <p className="text-gray-400 text-sm">No expert video available</p>
            )}
            {videoInfo && (
              <>
                <video
                  ref={videoRef}
                  key={videoInfo.url}
                  src={videoInfo.url}
                  muted
                  loop
                  className="w-full h-full object-cover"
                />
                {!started && (
                  <div className="absolute inset-0 flex items-center justify-center bg-black/30">
                    <div className="bg-white/90 rounded-full p-4">
                      <svg className="w-8 h-8 text-violet-600" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M8 5v14l11-7z" />
                      </svg>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        </div>
      </div>

      {/* AI Feedback */}
      <div className="bg-white border border-gray-200 rounded-2xl px-6 py-4 text-center min-h-[56px] flex items-center justify-center">
        <p className={feedback ? "text-gray-700 text-sm" : "text-gray-400 text-sm"}>
          {feedback ?? "AI Feedback will appear here..."}
        </p>
      </div>

      {/* Next sign */}
      {feedback && (
        <div className="flex justify-center mt-4">
          <button
            onClick={pickNextSign}
            className="bg-violet-600 hover:bg-violet-500 text-white text-sm font-semibold px-6 py-2.5 rounded-full transition-colors"
          >
            Next Sign
          </button>
        </div>
      )}
    </div>
  );
}
