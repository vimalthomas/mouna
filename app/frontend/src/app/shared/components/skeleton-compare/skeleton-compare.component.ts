import { Component, Input, OnChanges, OnDestroy } from '@angular/core';
import { NgTemplateOutlet } from '@angular/common';
import { ParameterResult, SkeletonFrame } from '../../../core/models/sign.model';

// MediaPipe Hands landmark topology -- fixed, not sign-specific. Fingers
// start at their OWN knuckle (MCP), not the wrist -- all 5 sharing the
// wrist as a start point made them fan out from one spot like claws. The
// wrist-to-knuckle gap is real palm, covered by the palm polygon instead.
const FINGER_CHAINS: Record<string, number[]> = {
  thumb: [1, 2, 3, 4],
  index: [5, 6, 7, 8],
  middle: [9, 10, 11, 12],
  ring: [13, 14, 15, 16],
  pinky: [17, 18, 19, 20],
};
// Palm outline: wrist + the five knuckle bases, in order across the hand --
// traces the actual palm shape instead of a circle that didn't match it.
const PALM_OUTLINE = [0, 1, 5, 9, 13, 17];

const HEAD_RADIUS = 0.34;
const FRAME_MS = 140;

@Component({
  selector: 'app-skeleton-compare',
  standalone: true,
  imports: [NgTemplateOutlet],
  template: `
    <div class="relative rounded-xl overflow-hidden bg-surface-sunk border border-border/50" [style.aspect-ratio]="mode === 'hands' ? '16/10' : '4/4.4'">
      <svg [attr.viewBox]="viewBox" class="w-full h-full">
        @if (frame) {
          @if (mode === 'body') {
            <polygon [attr.points]="torsoPoints(frame)" [attr.fill]="bodyColor" opacity="0.3"/>
            <circle [attr.cx]="headCenter(frame)[0]" [attr.cy]="headCenter(frame)[1]" [attr.r]="HEAD_RADIUS" [attr.fill]="bodyColor" opacity="0.3" [attr.stroke]="bodyColor" stroke-width="0.04"/>
            <g [attr.stroke]="locationColor" stroke-width="0.11" fill="none" stroke-linecap="round" stroke-linejoin="round">
              <polyline [attr.points]="armPoints(frame, 'l')"/>
              <polyline [attr.points]="armPoints(frame, 'r')"/>
            </g>
          }
          @if (frame.left_hand) { <ng-container [ngTemplateOutlet]="handTpl" [ngTemplateOutletContext]="{hand: frame.left_hand}"/> }
          @if (frame.right_hand) { <ng-container [ngTemplateOutlet]="handTpl" [ngTemplateOutletContext]="{hand: frame.right_hand}"/> }

          @if (frame.right_hand && frame.right_orientation) {
            <line [attr.x1]="frame.right_hand[0][0]" [attr.y1]="frame.right_hand[0][1]"
                  [attr.x2]="frame.right_hand[0][0] + frame.right_orientation[0]*0.7" [attr.y2]="frame.right_hand[0][1] + frame.right_orientation[1]*0.7"
                  [attr.stroke]="orientationColor" stroke-width="0.045" stroke-linecap="round"/>
          }
          @if (frame.left_hand && frame.left_orientation) {
            <line [attr.x1]="frame.left_hand[0][0]" [attr.y1]="frame.left_hand[0][1]"
                  [attr.x2]="frame.left_hand[0][0] + frame.left_orientation[0]*0.7" [attr.y2]="frame.left_hand[0][1] + frame.left_orientation[1]*0.7"
                  [attr.stroke]="orientationColor" stroke-width="0.045" stroke-linecap="round"/>
          }
        }

        <!-- reusable hand: real palm outline + tapered capsule fingers,
             widest at the knuckle, narrowing to the tip, with a small
             joint-circle at each segment start so the taper doesn't show
             a visible seam between segments. Chosen from a 6-way comparison
             after a thin-line version read as claws fanning from the wrist. -->
        <ng-template #handTpl let-hand="hand">
          <polygon [attr.points]="palmPoints(hand)" [attr.fill]="handshapeColor" opacity="0.3"/>
          @for (seg of fingerCapsules(hand); track $index) {
            <polygon [attr.points]="seg.poly" [attr.fill]="handshapeColor" opacity="0.85"/>
            <circle [attr.cx]="seg.jx" [attr.cy]="seg.jy" [attr.r]="seg.jr" [attr.fill]="handshapeColor" opacity="0.85"/>
          }
        </ng-template>
      </svg>

      @if (label) {
        <div class="absolute top-2 left-2 text-[10px] font-semibold uppercase tracking-wide text-ink-faint bg-surface/80 px-2 py-0.5 rounded">{{ label }}</div>
      }
      @if (!frame) {
        <div class="absolute inset-0 flex items-center justify-center text-ink-faint text-xs px-4 text-center">No skeleton data</div>
      }
    </div>
  `,
})
export class SkeletonCompareComponent implements OnChanges, OnDestroy {
  @Input() frames: SkeletonFrame[] | null = null;
  @Input() parameters: Record<string, ParameterResult> | null = null;
  @Input() label: string | null = null;
  @Input() animate = true;
  @Input() mode: 'body' | 'hands' = 'body';
  /** Base/idle color for this panel -- the only thing that should visually
   * distinguish one panel from another now that panels don't overlay each
   * other; mismatch coloring (amber) still overrides this per-segment. */
  @Input() color = '#3f5fc4';
  /** 0.5 = half speed, 1 = normal, 1.5+ = faster -- same control as the
   * Preview screen's video speed buttons, applied to the animation loop. */
  @Input() speedMultiplier = 1;

  private readonly BODY_VIEWBOX = '-2.6 -2.6 5.2 5.6';
  private readonly HAND_ZOOM_RADIUS = 1.1;

  readonly HEAD_RADIUS = HEAD_RADIUS;

  private progress = 0;
  private timer: ReturnType<typeof setInterval> | null = null;
  frame: SkeletonFrame | null = null;

  readonly warnColor = '#ab7a14';

  get bodyColor(): string { return this.color; }
  get locationColor(): string { return this.mismatch('location') ? this.warnColor : this.color; }
  get handshapeColor(): string { return this.mismatch('handshape') ? this.warnColor : this.color; }
  get orientationColor(): string { return this.mismatch('orientation') ? this.warnColor : this.color; }

  private mismatch(key: string): boolean {
    return this.parameters?.[key]?.match === false;
  }

  get viewBox(): string {
    if (this.mode === 'body') return this.BODY_VIEWBOX;
    const center = this.handsCenter();
    if (!center) return this.BODY_VIEWBOX;
    // Match the container's 16/10 aspect -- a square viewBox in a wider box
    // would just letterbox (empty bars either side) instead of actually
    // using the extra width to show the hand bigger.
    const rY = this.HAND_ZOOM_RADIUS;
    const rX = rY * 1.6;
    return `${center[0] - rX} ${center[1] - rY} ${rX * 2} ${rY * 2}`;
  }

  /** Tracks wherever the hand(s) actually are each frame, so "hands" mode
   * stays framed on the hand regardless of where the sign places it. */
  private handsCenter(): [number, number] | null {
    if (!this.frame) return null;
    const points = [this.frame.left_hand?.[0], this.frame.right_hand?.[0]].filter((p): p is [number, number] => !!p);
    if (points.length === 0) return null;
    const x = points.reduce((s, p) => s + p[0], 0) / points.length;
    const y = points.reduce((s, p) => s + p[1], 0) / points.length;
    return [x, y];
  }

  /** One tapered-quad + joint-circle per finger segment -- width shrinks
   * from base (knuckle) to tip so fingers read as fleshy, not wireframe.
   * Picked from a 6-option visual comparison. */
  fingerCapsules(hand: [number, number][]): { poly: string; jx: number; jy: number; jr: number }[] {
    const out: { poly: string; jx: number; jy: number; jr: number }[] = [];
    const BASE_WIDTH = 0.055;
    const TAPER = 0.18; // width shrinks ~18% per joint moving toward the tip
    for (const chain of Object.values(FINGER_CHAINS)) {
      for (let i = 0; i < chain.length - 1; i++) {
        const a = hand[chain[i]];
        const b = hand[chain[i + 1]];
        const wA = BASE_WIDTH * (1 - i * TAPER);
        const wB = BASE_WIDTH * (1 - (i + 1) * TAPER);
        const dx = b[0] - a[0];
        const dy = b[1] - a[1];
        const len = Math.hypot(dx, dy) || 1;
        const px = -dy / len;
        const py = dx / len;
        const poly = [
          [a[0] + px * wA, a[1] + py * wA],
          [b[0] + px * wB, b[1] + py * wB],
          [b[0] - px * wB, b[1] - py * wB],
          [a[0] - px * wA, a[1] - py * wA],
        ].map(([x, y]) => `${x},${y}`).join(' ');
        out.push({ poly, jx: a[0], jy: a[1], jr: wA });
      }
    }
    return out;
  }

  palmPoints(hand: [number, number][]): string {
    return PALM_OUTLINE.map(i => hand[i]).map(([x, y]) => `${x},${y}`).join(' ');
  }

  armPoints(frame: SkeletonFrame, side: 'l' | 'r'): string {
    const pts: [number, number][] = [frame.pose[`${side}_shoulder`], frame.pose[`${side}_elbow`], frame.pose[`${side}_wrist`]];
    return pts.map(([x, y]) => `${x},${y}`).join(' ');
  }

  torsoPoints(frame: SkeletonFrame): string {
    const pts: [number, number][] = [frame.pose['l_shoulder'], frame.pose['r_shoulder'], frame.pose['r_hip'], frame.pose['l_hip']];
    return pts.map(([x, y]) => `${x},${y}`).join(' ');
  }

  headCenter(frame: SkeletonFrame): [number, number] {
    const nose = frame.pose['nose'];
    return [nose[0], nose[1] - HEAD_RADIUS * 0.4]; // nudge up -- nose sits low in the face, not at head center
  }

  ngOnChanges(): void {
    if (this.timer) clearInterval(this.timer);
    this.progress = 0;
    this.updateFrame();
    const hasData = this.frames && this.frames.length > 0;
    if (!hasData || !this.animate) return;
    this.timer = setInterval(() => {
      this.progress = (this.progress + 1 / 24) % 1; // ~24 steps per loop, independent of array length
      this.updateFrame();
    }, FRAME_MS / Math.max(this.speedMultiplier, 0.1));
  }

  private updateFrame(): void {
    this.frame = this.frameAt(this.frames);
  }

  private frameAt(seq: SkeletonFrame[] | null | undefined): SkeletonFrame | null {
    if (!seq || seq.length === 0) return null;
    const idx = Math.min(seq.length - 1, Math.floor(this.progress * seq.length));
    return seq[idx];
  }

  ngOnDestroy(): void {
    if (this.timer) clearInterval(this.timer);
  }
}
