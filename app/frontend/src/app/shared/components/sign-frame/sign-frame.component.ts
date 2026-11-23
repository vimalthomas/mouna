import { AfterViewInit, ChangeDetectorRef, Component, ElementRef, Input, NgZone, OnDestroy, ViewChild, inject } from '@angular/core';
import { HandPath } from '../../../core/models/sign.model';

type Phase = 'start' | 'middle' | 'end' | 'full' | 'live';
type Pt = [number, number];

// Brand colors from the logo: coral and teal, one per hand.
const HAND_COLORS = ['#F48C77', '#75B4B4'];

/**
 * A frame (or the playing clip) of the sign's example video with each
 * hand's real path drawn on top -- the path comes from hand tracking on this
 * same video (scripts/pick_example_videos.py), so it lines up with the
 * signer. Phases:
 *   start  -> ring where the hand begins, path ahead shown faintly
 *   middle -> path so far, arrow at the current position
 *   end    -> whole path, arrowhead where the hand finishes
 *   full   -> the whole movement on one picture (middle frame)
 *   live   -> clip plays, a dot follows each hand along its path
 * The container takes the video's own aspect ratio, so normalized path
 * coords map straight onto the picture.
 */
@Component({
  selector: 'app-sign-frame',
  standalone: true,
  template: `
    <div class="relative rounded-xl overflow-hidden bg-black" [style.aspect-ratio]="aspect">
      @if (src) {
        <video #videoRef [src]="src" [muted]="true" playsinline preload="auto"
          [autoplay]="phase === 'live'" [loop]="false"
          class="absolute inset-0 w-full h-full object-fill"
          (loadedmetadata)="onMeta()" (timeupdate)="onTime()"></video>
      }
      @if (vw && paths.length) {
        <svg class="absolute inset-0 w-full h-full pointer-events-none" [attr.viewBox]="'0 0 ' + vw + ' ' + vh" preserveAspectRatio="none" aria-hidden="true">
          @for (p of drawn; track $index) {
            <!-- faint full path (where the hand will go) -->
            @if (p.faint.length > 1) {
              <polyline [attr.points]="pts(p.faint)" fill="none" stroke="rgba(0,0,0,0.35)" [attr.stroke-width]="sw * 2.2" stroke-linecap="round" stroke-linejoin="round" />
              <polyline [attr.points]="pts(p.faint)" fill="none" [attr.stroke]="p.color" stroke-opacity="0.45" [attr.stroke-width]="sw" stroke-linecap="round" stroke-linejoin="round" [attr.stroke-dasharray]="(sw * 1.6) + ' ' + (sw * 1.6)" />
            }
            <!-- solid path (where the hand has been) -->
            @if (p.solid.length > 1) {
              <polyline [attr.points]="pts(p.solid)" fill="none" stroke="rgba(0,0,0,0.45)" [attr.stroke-width]="sw * 2.4" stroke-linecap="round" stroke-linejoin="round" />
              <polyline [attr.points]="pts(p.solid)" fill="none" [attr.stroke]="p.color" [attr.stroke-width]="sw * 1.2" stroke-linecap="round" stroke-linejoin="round" />
            }
            @if (p.arrow) {
              <polygon [attr.points]="pts(p.arrow)" [attr.fill]="p.color" stroke="rgba(0,0,0,0.45)" [attr.stroke-width]="sw * 0.5" stroke-linejoin="round" />
            }
            @if (p.startRing) {
              <circle [attr.cx]="p.startRing[0]" [attr.cy]="p.startRing[1]" [attr.r]="sw * 4" fill="none" stroke="rgba(0,0,0,0.45)" [attr.stroke-width]="sw * 2" />
              <circle [attr.cx]="p.startRing[0]" [attr.cy]="p.startRing[1]" [attr.r]="sw * 4" fill="none" [attr.stroke]="p.color" [attr.stroke-width]="sw * 1.1" />
            }
            @if (p.stillRing) {
              <circle [attr.cx]="p.stillRing[0]" [attr.cy]="p.stillRing[1]" [attr.r]="sw * 7" fill="none" stroke="rgba(0,0,0,0.45)" [attr.stroke-width]="sw * 2" />
              <circle [attr.cx]="p.stillRing[0]" [attr.cy]="p.stillRing[1]" [attr.r]="sw * 7" fill="none" [attr.stroke]="p.color" [attr.stroke-width]="sw * 1.1" [attr.stroke-dasharray]="(sw * 2) + ' ' + (sw * 1.4)" />
              <text [attr.x]="p.stillRing[0]" [attr.y]="p.stillRing[1] - sw * 9" text-anchor="middle" [attr.font-size]="sw * 4.2" font-family="system-ui, sans-serif" font-weight="600" fill="white" stroke="rgba(0,0,0,0.55)" [attr.stroke-width]="sw * 0.9" paint-order="stroke">stays here</text>
            }
            @if (p.dot) {
              <circle [attr.cx]="p.dot[0]" [attr.cy]="p.dot[1]" [attr.r]="sw * 2.6" [attr.fill]="p.color" stroke="white" [attr.stroke-width]="sw * 0.8" />
            }
          }
        </svg>
      }
      @if (label) {
        <span class="absolute top-2 left-2 text-[11px] font-medium text-white/90 bg-black/45 px-2 py-0.5 rounded-md">{{ label }}</span>
      }
    </div>
  `,
})
export class SignFrameComponent implements AfterViewInit, OnDestroy {
  @Input() src: string | null = null;
  @Input() paths: HandPath[] = [];
  @Input() phase: Phase = 'full';
  @Input() label = '';
  @Input() playbackRate = 1;

  @ViewChild('videoRef') private videoRef?: ElementRef<HTMLVideoElement>;
  private cdr = inject(ChangeDetectorRef);
  private zone = inject(NgZone);
  private raf = 0;

  aspect = '16/10';
  vw = 0;
  vh = 0;
  /** Current time for drawing (seconds into the video). */
  private now = 0;

  get sw(): number { return Math.max(2, this.vw * 0.006); }

  /** Span covered by the hand paths (with a little lead-in / tail). */
  private get span(): [number, number] {
    const ts = this.paths.flatMap(p => [p.t[0], p.t[p.t.length - 1]]);
    if (!ts.length) return [0, 0];
    return [Math.min(...ts), Math.max(...ts)];
  }

  private targetTime(): number {
    const [a, b] = this.span;
    if (this.phase === 'start') return a;
    if (this.phase === 'end') return b;
    return (a + b) / 2;
  }

  ngAfterViewInit(): void {
    if (this.phase !== 'live') return;
    this.zone.runOutsideAngular(() => {
      const loop = () => {
        const v = this.videoRef?.nativeElement;
        if (v && !v.paused) { this.now = v.currentTime; this.cdr.detectChanges(); }
        this.raf = requestAnimationFrame(loop);
      };
      this.raf = requestAnimationFrame(loop);
    });
  }

  ngOnDestroy(): void { cancelAnimationFrame(this.raf); }

  onMeta(): void {
    const v = this.videoRef!.nativeElement;
    this.vw = v.videoWidth;
    this.vh = v.videoHeight;
    if (this.vw && this.vh) this.aspect = `${this.vw}/${this.vh}`;
    v.playbackRate = this.playbackRate;
    const [a, b] = this.span;
    if (this.phase === 'live') {
      v.currentTime = Math.max(0, a - 0.3);
      v.play().catch(() => {});
    } else {
      v.currentTime = this.targetTime();
      this.now = this.targetTime();
    }
    if (b <= a) this.now = v.currentTime;
  }

  /** Live mode loops just the signing span, with a short pause at the end. */
  onTime(): void {
    if (this.phase !== 'live') return;
    const v = this.videoRef!.nativeElement;
    const [a, b] = this.span;
    if (v.currentTime > b + 0.5 || v.ended) {
      v.currentTime = Math.max(0, a - 0.3);
      v.play().catch(() => {});
    }
  }

  get drawn(): { color: string; faint: Pt[]; solid: Pt[]; arrow: Pt[] | null; startRing: Pt | null; dot: Pt | null; stillRing?: Pt | null }[] {
    const W = this.vw, H = this.vh;
    return this.paths.map((p, i) => {
      const color = HAND_COLORS[i % HAND_COLORS.length];
      const all: Pt[] = p.x.map((x, k) => [x * W, p.y[k] * H]);
      if (p.still) {
        // Made in one spot: an arrow on a dot means nothing, so mark the place.
        const c: Pt = [all.reduce((a, q) => a + q[0], 0) / all.length, all.reduce((a, q) => a + q[1], 0) / all.length];
        return { color, faint: [], solid: [], arrow: null, startRing: null, dot: null, stillRing: c };
      }
      const t = this.phase === 'full' ? Infinity : this.phase === 'start' ? -Infinity : this.now;
      const upto = all.filter((_, k) => p.t[k] <= t);
      switch (this.phase) {
        case 'start':
          return { color, faint: all, solid: [], arrow: null, startRing: all[0] ?? null, dot: null };
        case 'middle':
          return { color, faint: all, solid: upto, arrow: arrowHead(upto, this.sw), startRing: null, dot: null };
        case 'end':
        case 'full':
          return { color, faint: [], solid: all, arrow: arrowHead(all, this.sw), startRing: all[0] ?? null, dot: null };
        case 'live': {
          const dot = upto.length ? upto[upto.length - 1] : (t < p.t[0] ? all[0] : all[all.length - 1]);
          return { color, faint: all, solid: upto, arrow: null, startRing: null, dot: dot ?? null };
        }
      }
    });
  }

  pts(points: Pt[]): string { return points.map(p => `${p[0]},${p[1]}`).join(' '); }
}

/** Arrowhead at the end of a path, pointing along its last stretch. */
function arrowHead(path: Pt[], sw: number): Pt[] | null {
  if (path.length < 2) return null;
  const end = path[path.length - 1];
  // look back far enough for a stable direction
  let k = path.length - 2;
  while (k > 0 && Math.hypot(end[0] - path[k][0], end[1] - path[k][1]) < sw * 3) k--;
  const dx = end[0] - path[k][0], dy = end[1] - path[k][1];
  const len = Math.hypot(dx, dy);
  if (len < 1e-6) return null;
  const ux = dx / len, uy = dy / len;
  const L = sw * 6, Wd = sw * 3.6;
  const base: Pt = [end[0] - ux * L * 0.6, end[1] - uy * L * 0.6];
  return [
    [end[0] + ux * L * 0.4, end[1] + uy * L * 0.4],
    [base[0] - uy * Wd, base[1] + ux * Wd],
    [base[0] + uy * Wd, base[1] - ux * Wd],
  ];
}
