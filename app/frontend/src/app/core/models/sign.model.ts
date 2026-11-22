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

export interface SkeletonFrame {
  pose: Record<string, [number, number]>;
  left_hand: [number, number][] | null;
  right_hand: [number, number][] | null;
  left_orientation: [number, number] | null;
  right_orientation: [number, number] | null;
  face?: FaceContours | null;
}

export interface ReferenceBuckets {
  handshape: string;
  location: string;
  orientation: string;
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

export interface GeminiValidation {
  valid: boolean;
  suggestion: string;
  explanation: string;
}

export interface PredictionResult {
  type: string;
  prediction: string;
  confidence: number;
  top3: Prediction[];
  uncertain: boolean;
  inference_time_ms: number | null;
  gemini_validation?: GeminiValidation;
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
}

export interface SignProgressDetailResponse {
  sign: string;
  total_attempts: number;
  correct_attempts: number;
  accuracy: number;
  completed: boolean;
  completed_at: string | null;
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
