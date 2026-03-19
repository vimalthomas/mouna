"use client";

import { useEffect, useRef, useState, useCallback } from "react";
import { InferenceResult } from "@/lib/types";

// Local type stubs for MediaPipe landmarks (avoids resolution issues with @mediapipe/tasks-vision)
interface MPLandmarker { detectForVideo(video: HTMLVideoElement, ts: number): unknown }


const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.33/wasm";
const POSE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task";
const HAND_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task";
const WS_URL = process.env.NEXT_PUBLIC_WS_URL ?? "ws://localhost:8000/ws/detect";
const RECORD_SECS = 3;
const FPS_TARGET = 25;

type Status = "loading" | "ready" | "recording" | "processing" | "error";

interface PoseLm { x: number; y: number; z: number }
interface HandResult {
  landmarks?: PoseLm[][];
  handedness?: { categoryName: string }[][];
}
interface PoseResult {
  landmarks?: PoseLm[][];
}

interface Props {
  onResult?: (result: InferenceResult) => void;
}

export default function Camera({ onResult }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [status, setStatus] = useState<Status>("loading");
  const [countdown, setCountdown] = useState(0);
  const [loadMsg, setLoadMsg] = useState("Initializing MediaPipe…");
  const [error, setError] = useState("");

  const poseLandmarkerRef = useRef<MPLandmarker | null>(null);
  const handLandmarkerRef = useRef<MPLandmarker | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const keypointBufferRef = useRef<number[][]>([]);
  const recordingRef = useRef(false);
  const rafRef = useRef<number>(0);
  const lastFrameTimeRef = useRef(0);

  const connectWS = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState <= 1) return;
    const ws = new WebSocket(WS_URL);
    ws.onmessage = (e) => {
      const data: InferenceResult = JSON.parse(e.data);
      onResult?.(data);
      setStatus("ready");
    };
    ws.onerror = () => {
      setError("Cannot reach backend. Is it running on :8000?");
      setStatus("error");
    };
    ws.onclose = () => { wsRef.current = null; };
    wsRef.current = ws;
  }, [onResult]);

  const extractKeypoints = useCallback(
    (poseResult: PoseResult, handResult: HandResult): number[] => {
      const kp = new Array(225).fill(0);
      const pose = poseResult.landmarks?.[0];
      if (pose) {
        for (let i = 0; i < Math.min(pose.length, 33); i++) {
          kp[i * 3] = pose[i].x;
          kp[i * 3 + 1] = pose[i].y;
          kp[i * 3 + 2] = pose[i].z;
        }
      }
      const hl = handResult.landmarks;
      const hd = handResult.handedness;
      if (hl && hd) {
        for (let h = 0; h < hl.length; h++) {
          const side = hd[h]?.[0]?.categoryName;
          const offset = side === "Left" ? 99 : 162;
          for (let j = 0; j < Math.min(hl[h].length, 21); j++) {
            kp[offset + j * 3] = hl[h][j].x;
            kp[offset + j * 3 + 1] = hl[h][j].y;
            kp[offset + j * 3 + 2] = hl[h][j].z;
          }
        }
      }
      return kp;
    },
    []
  );

  const processFrame = useCallback(
    (now: number) => {
      rafRef.current = requestAnimationFrame(processFrame);
      if (!videoRef.current || !poseLandmarkerRef.current || !handLandmarkerRef.current) return;
      if (now - lastFrameTimeRef.current < 1000 / FPS_TARGET) return;
      lastFrameTimeRef.current = now;
      const video = videoRef.current;
      if (video.readyState < 2) return;
      try {
        const pose = poseLandmarkerRef.current.detectForVideo(video, now) as unknown as PoseResult;
        const hand = handLandmarkerRef.current.detectForVideo(video, now) as unknown as HandResult;
        if (recordingRef.current) {
          keypointBufferRef.current.push(extractKeypoints(pose, hand));
        }
      } catch {
        // suppress stale timestamp errors
      }
    },
    [extractKeypoints]
  );

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        // @ts-ignore – mediapipe exports are resolved at runtime via CDN WASM
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        const mp = await import("@mediapipe/tasks-vision") as any;
        const { FilesetResolver, PoseLandmarker, HandLandmarker } = mp;

        setLoadMsg("Loading pose model…");
        const vision = await FilesetResolver.forVisionTasks(WASM_URL);

        const pose = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: POSE_MODEL_URL, delegate: "GPU" },
          runningMode: "VIDEO",
          numPoses: 1,
        });
        setLoadMsg("Loading hand model…");
        const hand = await HandLandmarker.createFromOptions(vision, {
          baseOptions: { modelAssetPath: HAND_MODEL_URL, delegate: "GPU" },
          runningMode: "VIDEO",
          numHands: 2,
        });

        if (cancelled) return;
        poseLandmarkerRef.current = pose;
        handLandmarkerRef.current = hand;

        setLoadMsg("Starting camera…");
        const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
        if (cancelled) { stream.getTracks().forEach((t) => t.stop()); return; }
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }

        rafRef.current = requestAnimationFrame(processFrame);
        connectWS();
        setStatus("ready");
      } catch (err) {
        if (!cancelled) {
          setError(String(err));
          setStatus("error");
        }
      }
    })();
    return () => {
      cancelled = true;
      cancelAnimationFrame(rafRef.current);
      wsRef.current?.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const startRecording = useCallback(() => {
    if (status !== "ready") return;
    connectWS();
    keypointBufferRef.current = [];
    recordingRef.current = true;
    setStatus("recording");
    setCountdown(RECORD_SECS);

    const tick = setInterval(() => {
      setCountdown((c) => {
        if (c <= 1) clearInterval(tick);
        return c - 1;
      });
    }, 1000);

    setTimeout(() => {
      recordingRef.current = false;
      clearInterval(tick);
      const frames = keypointBufferRef.current;
      if (!frames.length || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        setStatus("ready");
        return;
      }
      setStatus("processing");
      wsRef.current.send(JSON.stringify({ keypoints: frames }));
    }, RECORD_SECS * 1000);
  }, [status, connectWS]);

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
            <p className="text-zinc-400 text-sm">{loadMsg}</p>
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
          : "Record Sign  (2s)"}
      </button>
    </div>
  );
}
