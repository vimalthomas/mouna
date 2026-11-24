export interface Sign {
  word: string;
  category: string;
  difficulty: string;
  description: string;
  manual_checks: string[];
  nms_required: boolean;
  nms_type: string | null;
  nms_notes: string | null;
}

export type ManualCheckStep = 'handshape' | 'location' | 'orientation' | 'hand_count';
export type PrepStep = ManualCheckStep | 'facial';

export interface BBox {
  x_min: number;
  x_max: number;
  y_min: number;
  y_max: number;
}

export interface ParameterResult {
  target: string | null;
  yours: string | null;
  match: boolean | null;
  why?: string | null;
  /** Plain wording of target/yours buckets; null where not trustworthy (location). */
  target_plain?: string | null;
  yours_plain?: string | null;
  /** v4: false when this check failed the judgement benchmark (shown as a hint, not a mistake). */
  reliable?: boolean;
}

export interface FaceContours {
  face_oval: [number, number][];
  left_eye: [number, number][];
  right_eye: [number, number][];
  left_eyebrow: [number, number][];
  right_eyebrow: [number, number][];
  lips_outer: [number, number][];
  lips_inner: [number, number][];
}

/** v3 3D skin: per frame of a clip, both hands (video pixels + depth), each
 * hand's 3D shape, upper-body pose and the face mesh. From GET
 * /signs/{word}/skin (example videos) or CoachingMessage.attempt_skin. */
export interface SkinFrame {
  h: ([number, number, number][] | null)[];
  w: ([number, number, number][] | null)[];
  p: [number, number, number][] | null;
  f: [number, number, number][] | null;
}
export interface SkinData { width: number; height: number; fps: number; n_frames: number; frames: SkinFrame[]; mirrored?: boolean; }

export interface SkeletonFrame {
  pose: Record<string, [number, number]>;
  left_hand: [number, number][] | null;
  right_hand: [number, number][] | null;
  left_orientation: [number, number] | null;
  right_orientation: [number, number] | null;
  face?: FaceContours | null;
  /** 0..1 position of this frame within the source buffer/clip. */
  t?: number;
  /** Inverse of the body-relative normalization in the source image's
   * normalized coords: image_xy = xy * scale + origin. */
  origin?: [number, number];
  scale?: number;
  /** Hand-local 3D (wrist-centered, wrist->middle-knuckle scaled). */
  left_hand_3d?: [number, number, number][] | null;
  right_hand_3d?: [number, number, number][] | null;
}

export interface ReferenceBuckets {
  handshape: string;
  location: string;
  orientation: string;
  /** Imperative instructions per parameter; null where not trustworthy (location). */
  howto?: Partial<Record<'handshape' | 'location' | 'orientation', string | null>>;
  plain?: Partial<Record<'handshape' | 'location' | 'orientation', string | null>>;
  /** When the hands are up in the example video shown (seconds). */
  hands_up_seconds?: { start: number; end: number } | null;
  /** Each hand's path through the example video, normalized image coords. */
  hand_paths?: HandPath[];
  hand_bbox: BBox | null;
}

export interface StepResult {
  type: 'step_result';
  step: PrepStep;
  applicable: boolean;
  message?: string;
  // manual steps
  target?: string | null;
  yours?: string | null;
  match?: boolean | null;
  why?: string | null;
  target_plain?: string | null;
  yours_plain?: string | null;
  /** Approved signers split on this part, so it isn't graded for this sign. */
  uncertain?: boolean;
  // facial step
  nms_type?: string;
  present?: boolean;
  detected?: boolean | null;
  reference_detected?: boolean | null;
  validated?: boolean;
  notes?: string[];
}

export interface SignListResponse {
  signs: Sign[];
}

export interface VideoResponse {
  word: string;
  video_url: string;
  frame_start: number | null;
  frame_end: number | null;
  fps: number | null;
}

export interface Prediction {
  sign: string;
  confidence: number;
}

export interface PredictionResult {
  type: string;
  prediction: string;
  confidence: number;
  top3: Prediction[];
  uncertain: boolean;
  inference_time_ms: number | null;
}

export type KeypointFrame = number[];

export interface SignProgress {
  total_attempts: number;
  correct_attempts: number;
  accuracy: number;
  completed: boolean;
}

export interface CoachingMessage {
  type: string;
  prediction?: string;
  confidence?: number;
  correct?: boolean;
  verdict?: 'good' | 'needs_work' | 'wrong_sign';
  embedding_similarity?: number | null;
  hand_count_match?: boolean | null;
  parameters?: Record<string, ParameterResult> | null;
  feedback?: string;
  /** Which AI wrote the message ("claude" / "gemini"), or "template" without a key. */
  feedback_source?: 'claude' | 'gemini' | 'template';
  /** v3: the learner's own hands and face as 3D skin (mirrored like the camera preview). */
  attempt_skin?: SkinData | null;
  /** v4: per check, how it scored on the judgement benchmark. */
  check_reliability?: Record<string, { false_alarm: number | null; catches: number | null; reliable: boolean }>;
  parameter_scores?: Record<string, ParameterScore> | null;
  charts?: AttemptCharts | null;
  /** False when the sign has no reference recording, so nothing was compared. */
  graded?: boolean;
  tips?: string[];
  attempt_number?: number;
  session_accuracy?: number;
  session_id?: string;
  message?: string;
  supported?: boolean;
  sign_completed?: boolean;
  sign_progress?: SignProgress;
  frame_count?: number;
  pose_detected?: boolean;
  left_hand_detected?: boolean;
  right_hand_detected?: boolean;
  face_detected?: boolean;
  face_bbox?: BBox | null;
  hand_bbox?: BBox | null;
  attempt_skeleton_raw?: SkeletonFrame[] | null;
  attempt_skeleton_canonical?: SkeletonFrame[] | null;
  /** True when scoring mirrored the attempt (left-dominant signer). */
  attempt_mirrored?: boolean;
  reference_skeleton?: SkeletonFrame[] | null;
  divergent_snapshots?: Record<string, SkeletonFrame> | null;
  divergent_frames?: Record<string, number> | null;
  looks_like?: string | null;
  looks_like_skeleton?: SkeletonFrame[] | null;
}

export interface WeakSign {
  sign: string;
  accuracy: number;
}

export interface ReviewDueSign {
  sign: string;
  accuracy: number;
  days_since_practiced: number;
  overdue_days: number;
}

export interface SignProgressDetail {
  sign: string;
  totalAttempts: number;
  correctAttempts: number;
  accuracy: number;
  completed: boolean;
  completedAt: string | null;
  /** Last up-to-5 results, oldest first (learned = 3+ correct). */
  recent: boolean[];
}

export interface SignProgressDetailResponse {
  sign: string;
  total_attempts: number;
  correct_attempts: number;
  accuracy: number;
  completed: boolean;
  completed_at: string | null;
  recent?: boolean[];
}

export interface UserStats {
  totalSessions: number;
  signsLearned: number;
  accuracy: number;
  streak: number;
  weakSigns: WeakSign[];
  signs: SignProgressDetail[];
}

export interface UserStatsResponse {
  total_sessions: number;
  signs_learned: number;
  accuracy: number;
  streak: number;
  weak_signs: WeakSign[];
  signs: SignProgressDetailResponse[];
}

export interface AttemptCharts {
  fingers: string[];
  finger_openness: { you: number[]; target: number[] };
  travel: { you: number[]; target: number[] };
  brow: { you: number[] | null; target: number[] | null; threshold: number } | null;
}

export interface HandPath {
  /** Screen side ('screen_left'/'screen_right'), or 'only' for one hand. */
  hand: string;
  /** Made in one spot (tiny travel): drawn as a "stays here" ring, no arrow. */
  still?: boolean;
  t: number[];
  x: number[];
  y: number[];
}

export interface SignCatalogEntry {
  in_asl_lex: boolean;
  in_cdi: boolean;
  /** ASL-CDI semantic category, or "Other signs". */
  category: string;
  /** ASL-LEX 2.0 frequency rating by deaf signers (1-7). */
  frequency: number | null;
  /** On ASL-CDI 2.0 and ASL-LEX frequency >= 5.0. */
  essential: boolean;
  in_model_vocab: boolean;
}

/** One of the five parameters, scored against fluent signers (src/mouna/coach/param_scores.py). */
export interface ParameterScore {
  label: string;
  level: 'fluent' | 'close' | 'needs_work' | null;
  score: number | null;
  reliable: boolean;
  method: 'vector+cp' | 'cp' | 'none';
  evidence: { dim: string; yours: number; fluent: [number, number]; outside: number; unit: string }[];
  cp_worst: string | null;
  ratio?: number;
  n_fluent?: number;
}
