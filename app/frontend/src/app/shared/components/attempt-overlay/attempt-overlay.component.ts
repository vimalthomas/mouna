import { AfterViewInit, Component, ElementRef, Input, NgZone, OnChanges, OnDestroy, ViewChild, inject } from '@angular/core';
import { ParameterResult, SkeletonFrame } from '../../../core/models/sign.model';

// MediaPipe hand topology: finger chains start at their own knuckle; the
// wrist-to-knuckle span is drawn as the palm outline.
const FINGER_CHAINS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]];
const PALM_OUTLINE = [0, 1, 5, 9, 13, 17, 0];

const GOOD = '#22c55e';
const WARN = '#f59e0b';
const GHOST = 'rgba(255,255,255,0.8)';

type Pt = [number, number];

/**
 * Draws on the user's own recorded attempt video:
 *  - the TARGET sign's arms + hands as a translucent ghost, placed in the
 *    user's own body frame (their shoulders/scale) and time-stretched over
 *    the part of the clip where their hands were actually up;
 *  - the user's OWN tracked hands on top, green when handshape matched,
 *    amber when it didn't;
 *  - a target ring + arrow when location mismatched, and palm-direction
 *    arrows when orientation mismatched.
 *
 * Coordinates: skeleton frames are body-relative; each carries `origin` and
 * `scale` so image_xy = rel * scale + origin (normalized 0..1 of the full
 * captured frame, which is also the full recorded video frame).
 */
@Component({
  selector: 'app-attempt-overlay',
  standalone: true,
  template: `
    <div class="relative rounded-xl overflow-hidden bg-black border border-border/50" [style.aspect-ratio]="aspect">
      @if (videoUrl) {
        <div class="absolute inset-0" style="transform: scaleX(-1);">
          <video #videoRef [src]="videoUrl" autoplay loop [muted]="true" playsinline class="w-full h-full block object-fill"
                 (loadedmetadata)="onMeta()"></video>
          <canvas #canvasRef class="absolute inset-0 w-full h-full pointer-events-none"></canvas>
        </div>
      } @else {
        <div class="absolute inset-0 flex items-center justify-center text-white/60 text-sm px-4 text-center">No recording of this attempt available in this browser.</div>
      }
      @if (videoUrl && notice) {
        <div class="absolute bottom-2 left-2 right-2 text-[11px] text-white bg-black/60 px-2 py-1 rounded">{{ notice }}</div>
      }
      @if (label) {
        <div class="absolute top-2 left-2 text-[11px] font-medium text-white/90 bg-black/45 px-2 py-0.5 rounded-md">{{ label }}</div>
      }
    </div>
    <div class="flex flex-wrap items-center gap-x-4 gap-y-1 mt-2 text-[11px] text-ink-muted">
      <label class="flex items-center gap-1.5 cursor-pointer select-none">
        <input type="checkbox" [checked]="showGhost" (change)="showGhost = !showGhost" />
        <span class="inline-block w-3 h-0.5 bg-ink-muted"></span> Target (ghost)
      </label>
      <label class="flex items-center gap-1.5 cursor-pointer select-none">
        <input type="checkbox" [checked]="showMine" (change)="showMine = !showMine" />
        <span class="inline-block w-2 h-2 rounded-full" [style.background]="handColor"></span> Your hands
      </label>
      @if (mismatch('location')) { <span class="flex items-center gap-1"><span class="inline-block w-2.5 h-2.5 rounded-full border border-dashed" [style.border-color]="WARN"></span> where your hand should be</span> }
      @if (mismatch('orientation')) { <span>arrows = which way the palm faces</span> }
    </div>
  `,
})
export class AttemptOverlayComponent implements AfterViewInit, OnChanges, OnDestroy {
  @Input() videoUrl: string | null = null;
  /** attempt_skeleton_raw -- the same un-mirrored frames the video holds. */
  @Input() attemptFrames: SkeletonFrame[] | null = null;
  @Input() referenceFrames: SkeletonFrame[] | null = null;
  @Input() parameters: Record<string, ParameterResult> | null = null;
  /** True when scoring mirrored the attempt (left-dominant signer); the
   * right-handed reference ghost is then mirrored to match their hands. */
  @Input() mirrored = false;
  @Input() speedMultiplier = 1;
  /** Fallback when the browser reports MediaRecorder webm duration as Infinity. */
  @Input() fallbackDurationSec = 5;
  @Input() label: string | null = null;

  @ViewChild('videoRef') private videoRef?: ElementRef<HTMLVideoElement>;
  @ViewChild('canvasRef') private canvasRef?: ElementRef<HTMLCanvasElement>;

  private zone = inject(NgZone);
  private raf = 0;
  aspect = '4/3';
  showGhost = true;
  showMine = true;
  readonly WARN = WARN;

  /** Says why something isn't drawn, instead of silently showing bare video. */
  get notice(): string {
    if (!this.attemptFrames || this.attemptFrames.length === 0) return "I couldn't track your hands in this recording. Keep both hands in frame with good light and try again.";
    if (!this.referenceFrames || this.referenceFrames.length === 0) return 'No target ghost for this sign: there is no reference recording yet.';
    return '';
  }

  get handColor(): string { return this.mismatch('handshape') ? WARN : GOOD; }

  mismatch(key: string): boolean { return this.parameters?.[key]?.match === false; }

  ngAfterViewInit(): void {
    this.zone.runOutsideAngular(() => {
      const loop = () => { this.draw(); this.raf = requestAnimationFrame(loop); };
      this.raf = requestAnimationFrame(loop);
    });
  }

  ngOnChanges(): void {
    const v = this.videoRef?.nativeElement;
    if (v) v.playbackRate = this.speedMultiplier;
  }

  onMeta(): void {
    const v = this.videoRef!.nativeElement;
    if (v.videoWidth && v.videoHeight) this.aspect = `${v.videoWidth}/${v.videoHeight}`;
    v.playbackRate = this.speedMultiplier;
  }

  ngOnDestroy(): void { cancelAnimationFrame(this.raf); }

  // ---- timing ----

  private videoProgress(): number {
    const v = this.videoRef?.nativeElement;
    if (!v) return 0;
    const dur = Number.isFinite(v.duration) && v.duration > 0 ? v.duration : this.fallbackDurationSec;
    return Math.min(1, Math.max(0, v.currentTime / dur));
  }

  /** Interpolated frame at buffer-time t. Outside the sequence's own time
   * span returns null (hands weren't detected there) unless `clamp`. */
  private frameAt(seq: SkeletonFrame[], t: number, clamp: boolean): SkeletonFrame | null {
    if (seq.length === 0) return null;
    const t0 = seq[0].t ?? 0;
    const t1 = seq[seq.length - 1].t ?? 1;
    if (t < t0 || t > t1) {
      if (!clamp) return null;
      return t < t0 ? seq[0] : seq[seq.length - 1];
    }
    let i = 0;
    while (i < seq.length - 2 && (seq[i + 1].t ?? 0) < t) i++;
    const a = seq[i];
    const b = seq[Math.min(i + 1, seq.length - 1)];
    const ta = a.t ?? 0;
    const tb = b.t ?? 0;
    const k = tb > ta ? (t - ta) / (tb - ta) : 0;
    return lerpFrame(a, b, Math.min(1, Math.max(0, k)));
  }

  // ---- drawing ----

  private draw(): void {
    const canvas = this.canvasRef?.nativeElement;
    if (!canvas) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const W = Math.round(canvas.clientWidth * dpr);
    const H = Math.round(canvas.clientHeight * dpr);
    if (canvas.width !== W || canvas.height !== H) { canvas.width = W; canvas.height = H; }
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, W, H);

    const attempt = this.attemptFrames;
    if (!attempt || attempt.length === 0 || attempt[0].origin === undefined) return;

    const p = this.videoProgress();
    const mine = this.frameAt(attempt, p, false);
    // Body frame for the ghost: the user's own, clamped so the ghost stays
    // anchored to their shoulders before/after their hands come up.
    const body = mine ?? this.frameAt(attempt, p, true)!;
    const origin = body.origin!;
    const scale = body.scale ?? 1;
    const px = (rel: Pt, mirror = false): Pt => [((mirror ? -rel[0] : rel[0]) * scale + origin[0]) * W, (rel[1] * scale + origin[1]) * H];
    const lw = Math.max(2, Math.min(9, scale * W * 0.035));

    // Ghost: map the reference's whole sign onto the attempt's active span.
    let ghost: SkeletonFrame | null = null;
    const ref = this.referenceFrames;
    if (this.showGhost && ref && ref.length > 0) {
      const a0 = attempt[0].t ?? 0;
      const a1 = attempt[attempt.length - 1].t ?? 1;
      const q = a1 > a0 ? Math.min(1, Math.max(0, (p - a0) / (a1 - a0))) : 0;
      const r0 = ref[0].t ?? 0;
      const r1 = ref[ref.length - 1].t ?? 1;
      ghost = this.frameAt(ref, r0 + q * (r1 - r0), true);
      if (ghost) this.drawFigure(ctx, ghost, pt => px(pt, this.mirrored), GHOST, lw, true);
    }

    if (this.showMine && mine) {
      this.drawHands(ctx, mine, pt => px(pt), this.handColor, lw * 0.8);
    }

    if (ghost && mine && this.mismatch('location')) {
      const target = handCenter(ghost);
      const yours = handCenter(mine);
      if (target && yours) {
        const T = px(target, this.mirrored);
        const Y = px(yours);
        ctx.save();
        ctx.strokeStyle = WARN;
        ctx.lineWidth = lw * 0.6;
        ctx.setLineDash([lw * 1.2, lw]);
        ctx.beginPath();
        ctx.arc(T[0], T[1], scale * W * 0.22, 0, Math.PI * 2);
        ctx.stroke();
        ctx.setLineDash([]);
        arrow(ctx, Y, T, WARN, lw * 0.6);
        ctx.restore();
      }
    }

    if (this.mismatch('orientation')) {
      const LEN = 0.5; // body-relative units (shoulder widths)
      const tip = (h: Pt, o: Pt): Pt => [h[0] + o[0] * LEN, h[1] + o[1] * LEN];
      if (mine) for (const [hand, o] of [[mine.left_hand, mine.left_orientation], [mine.right_hand, mine.right_orientation]] as const) {
        if (hand && o) arrow(ctx, px(hand[0]), px(tip(hand[0], o)), WARN, lw * 0.6);
      }
      if (ghost) for (const [hand, o] of [[ghost.left_hand, ghost.left_orientation], [ghost.right_hand, ghost.right_orientation]] as const) {
        if (hand && o) arrow(ctx, px(hand[0], this.mirrored), px(tip(hand[0], o), this.mirrored), GHOST, lw * 0.6);
      }
    }
  }

  private drawFigure(ctx: CanvasRenderingContext2D, f: SkeletonFrame, P: (p: Pt) => Pt, color: string, lw: number, ghost: boolean): void {
    ctx.save();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    for (const side of ['l', 'r']) {
      const chain = [f.pose[`${side}_shoulder`], f.pose[`${side}_elbow`], f.pose[`${side}_wrist`]].filter(Boolean).map(P);
      strokeOutlined(ctx, chain, color, lw * 1.4, ghost);
    }
    ctx.restore();
    this.drawHands(ctx, f, P, color, lw, ghost);
  }

  private drawHands(ctx: CanvasRenderingContext2D, f: SkeletonFrame, P: (p: Pt) => Pt, color: string, lw: number, ghost = false): void {
    ctx.save();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    for (const hand of [f.left_hand, f.right_hand]) {
      if (!hand) continue;
      const pts = hand.map(p => P(p as Pt));
      // palm fill
      ctx.beginPath();
      PALM_OUTLINE.forEach((i, k) => k === 0 ? ctx.moveTo(pts[i][0], pts[i][1]) : ctx.lineTo(pts[i][0], pts[i][1]));
      ctx.fillStyle = ghost ? 'rgba(255,255,255,0.18)' : hexAlpha(color, 0.28);
      ctx.fill();
      strokeOutlined(ctx, PALM_OUTLINE.map(i => pts[i]), color, lw * 0.7, ghost);
      for (const chain of FINGER_CHAINS) strokeOutlined(ctx, chain.map(i => pts[i]), color, lw * 0.75, ghost);
      if (!ghost) {
        ctx.fillStyle = color;
        for (const q of pts) { ctx.beginPath(); ctx.arc(q[0], q[1], lw * 0.45, 0, Math.PI * 2); ctx.fill(); }
      }
    }
    ctx.restore();
  }
}

// ---- helpers ----

function strokeOutlined(ctx: CanvasRenderingContext2D, pts: Pt[], color: string, lw: number, ghost: boolean): void {
  if (pts.length < 2) return;
  const path = () => { ctx.beginPath(); pts.forEach((p, i) => i === 0 ? ctx.moveTo(p[0], p[1]) : ctx.lineTo(p[0], p[1])); };
  // Dark halo keeps lines legible over any background in the video.
  path(); ctx.strokeStyle = 'rgba(0,0,0,0.45)'; ctx.lineWidth = lw + 3; ctx.stroke();
  path(); ctx.strokeStyle = color; ctx.lineWidth = ghost ? lw * 0.9 : lw; ctx.stroke();
}

function arrow(ctx: CanvasRenderingContext2D, from: Pt, to: Pt, color: string, lw: number): void {
  const dx = to[0] - from[0];
  const dy = to[1] - from[1];
  const len = Math.hypot(dx, dy);
  if (len < 4) return;
  const ux = dx / len;
  const uy = dy / len;
  const head = Math.min(len * 0.35, lw * 5);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = lw;
  ctx.lineCap = 'round';
  ctx.beginPath(); ctx.moveTo(from[0], from[1]); ctx.lineTo(to[0] - ux * head * 0.8, to[1] - uy * head * 0.8); ctx.stroke();
  ctx.beginPath();
  ctx.moveTo(to[0], to[1]);
  ctx.lineTo(to[0] - ux * head - uy * head * 0.5, to[1] - uy * head + ux * head * 0.5);
  ctx.lineTo(to[0] - ux * head + uy * head * 0.5, to[1] - uy * head - ux * head * 0.5);
  ctx.closePath(); ctx.fill();
  ctx.restore();
}

function hexAlpha(hex: string, a: number): string {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

function handCenter(f: SkeletonFrame): Pt | null {
  const w = [f.left_hand?.[0], f.right_hand?.[0]].filter((p): p is Pt => !!p);
  if (w.length === 0) return null;
  return [w.reduce((s, p) => s + p[0], 0) / w.length, w.reduce((s, p) => s + p[1], 0) / w.length];
}

function lerpPt(a: Pt, b: Pt, k: number): Pt { return [a[0] + (b[0] - a[0]) * k, a[1] + (b[1] - a[1]) * k]; }

function lerpList(a: Pt[] | null, b: Pt[] | null, k: number): Pt[] | null {
  if (a && b) return a.map((p, i) => lerpPt(p, b[i], k));
  return k < 0.5 ? a : b;
}

function lerpFrame(a: SkeletonFrame, b: SkeletonFrame, k: number): SkeletonFrame {
  const pose: Record<string, Pt> = {};
  for (const key of Object.keys(a.pose)) pose[key] = b.pose[key] ? lerpPt(a.pose[key], b.pose[key], k) : a.pose[key];
  return {
    ...a,
    pose,
    left_hand: lerpList(a.left_hand, b.left_hand, k),
    right_hand: lerpList(a.right_hand, b.right_hand, k),
    left_orientation: a.left_orientation && b.left_orientation ? lerpPt(a.left_orientation, b.left_orientation, k) : (k < 0.5 ? a.left_orientation : b.left_orientation),
    right_orientation: a.right_orientation && b.right_orientation ? lerpPt(a.right_orientation, b.right_orientation, k) : (k < 0.5 ? a.right_orientation : b.right_orientation),
    origin: a.origin && b.origin ? lerpPt(a.origin, b.origin, k) : a.origin,
    scale: (a.scale ?? 1) + ((b.scale ?? 1) - (a.scale ?? 1)) * k,
  };
}
