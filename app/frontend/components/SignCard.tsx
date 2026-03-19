"use client";

import { useRef, useState } from "react";
import { Sign } from "@/lib/types";

const difficultyColors: Record<string, string> = {
  easy: "bg-emerald-900 text-emerald-300",
  medium: "bg-amber-900 text-amber-300",
  hard: "bg-red-900 text-red-300",
};

interface VideoInfo {
  url: string;
  start: number | null;
  end: number | null;
}

interface Props {
  sign: Sign;
  highlight?: boolean;
}

export default function SignCard({ sign, highlight = false }: Props) {
  const [videoInfo, setVideoInfo] = useState<VideoInfo | null | undefined>(undefined);
  const [expanded, setExpanded] = useState(false);
  const videoRef = useRef<HTMLVideoElement>(null);

  async function handleExpand() {
    if (expanded) { setExpanded(false); return; }
    setExpanded(true);
    if (videoInfo === undefined) {
      try {
        const res = await fetch(`http://localhost:8000/video/${sign.word}`);
        const data = await res.json();
        setVideoInfo(data.url ? { url: data.url, start: data.start, end: data.end } : null);
      } catch {
        setVideoInfo(null);
      }
    }
  }

  function handleLoadedMetadata(info: VideoInfo) {
    const vid = videoRef.current;
    if (!vid) return;
    const start = info.start ?? 0;
    vid.currentTime = start;
    vid.play().catch(() => {});
  }

  function handleTimeUpdate(info: VideoInfo) {
    const vid = videoRef.current;
    if (!vid || info.end == null) return;
    if (vid.currentTime >= info.end) {
      vid.currentTime = info.start ?? 0;
    }
  }

  return (
    <div
      className={`rounded-xl border transition-colors cursor-pointer ${
        highlight
          ? "border-violet-500 bg-violet-950"
          : "border-zinc-800 bg-zinc-900 hover:border-zinc-600"
      }`}
      onClick={handleExpand}
    >
      <div className="p-4">
        <div className="flex items-start justify-between gap-2 mb-2">
          <h3 className="text-white font-semibold capitalize text-lg leading-tight">
            {sign.word}
          </h3>
          <div className="flex items-center gap-1.5 shrink-0">
            <span
              className={`text-xs font-medium px-2 py-0.5 rounded-full capitalize ${difficultyColors[sign.difficulty]}`}
            >
              {sign.difficulty}
            </span>
            <span className="text-zinc-500 text-xs">{expanded ? "▲" : "▶"}</span>
          </div>
        </div>
        <p className="text-zinc-400 text-sm leading-relaxed">{sign.description}</p>
        <p className="mt-2 text-xs text-zinc-600 capitalize">{sign.category}</p>
      </div>

      {expanded && (
        <div className="border-t border-zinc-800 p-3" onClick={(e) => e.stopPropagation()}>
          {videoInfo === undefined && (
            <div className="flex items-center justify-center h-24 gap-2 text-zinc-500 text-sm">
              <div className="w-4 h-4 border-2 border-zinc-600 border-t-transparent rounded-full animate-spin" />
              Loading…
            </div>
          )}
          {videoInfo === null && (
            <p className="text-zinc-600 text-xs text-center py-4">No sample video available</p>
          )}
          {videoInfo && (
            <video
              ref={videoRef}
              key={videoInfo.url}
              src={`http://localhost:8000${videoInfo.url}`}
              controls
              muted
              className="w-full rounded-lg bg-black"
              style={{ maxHeight: "200px" }}
              onLoadedMetadata={() => handleLoadedMetadata(videoInfo)}
              onTimeUpdate={() => handleTimeUpdate(videoInfo)}
            />
          )}
        </div>
      )}
    </div>
  );
}
