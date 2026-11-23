import { AfterViewInit, Component, ElementRef, Input, NgZone, OnChanges, OnDestroy, SimpleChanges, ViewChild, inject } from '@angular/core';
import { environment } from '../../../../environments/environment';
import { SkinData } from '../../../core/models/sign.model';

const HAND_EDGES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]];
const FINGER = (a: number, b: number) => ([5, 9, 13, 17].includes(a) && [9, 13, 17].includes(b)) || (a === 0 && b !== 1) ? 0 : 1 + Math.floor((Math.max(a, b) - 1) / 4);
const FINGER_COLORS = ['#c9ccd3', '#e8a33d', '#2a78d6', '#1f9e89', '#c2477f', '#7a6cd0'];  // palm, thumb..pinky
const ARM_EDGES = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16]];
const FACE_PARTS = ['FACE_LANDMARKS_FACE_OVAL', 'FACE_LANDMARKS_LIPS', 'FACE_LANDMARKS_LEFT_EYEBROW', 'FACE_LANDMARKS_RIGHT_EYEBROW',
  'FACE_LANDMARKS_LEFT_EYE', 'FACE_LANDMARKS_RIGHT_EYE'];

let faceConn: Promise<Record<string, number[][]>> | null = null;

/**
 * A video with its own tracked skeleton drawn on top, frame by frame: hands
 * (coloured by finger), shoulders and arms, and the face outline (brows,
 * eyes, lips). The skeleton comes from that same video's tracking (skin
 * data), so it lines up with the person in it. `word` loads a sign's example
 * skin; `skin` passes one directly (the learner's attempt, already mirrored,
 * shown over a mirrored video).
 */
@Component({
  selector: 'app-video-bones',
  standalone: true,
  template: `
    <div class="relative rounded-xl overflow-hidden bg-black border border-border/50" style="aspect-ratio: 4/3;">
      @if (src) {
        <video #videoRef [src]="src" autoplay loop [muted]="true" playsinline class="absolute inset-0 w-full h-full object-contain"
          [style.transform]="mirror ? 'scaleX(-1)' : ''"></video>
      } @else {
        <div class="absolute inset-0 flex items-center justify-center text-white/60 text-xs">{{ emptyText }}</div>
      }
      <canvas #canvasRef class="absolute inset-0 w-full h-full pointer-events-none"></canvas>
      <span class="absolute top-2 left-2 text-[11px] font-medium text-white/90 bg-black/45 px-2 py-0.5 rounded-md">{{ label }}</span>
    </div>
  `,
})
export class VideoBonesComponent implements AfterViewInit, OnChanges, OnDestroy {
  @Input() src: string | null = null;
  @Input() word = '';
  @Input() skin: SkinData | null = null;
  @Input() mirror = false;
  @Input() showBones = true;
  @Input() speed = 1;
  @Input() label = '';
  @Input() emptyText = 'No video';
  @ViewChild('videoRef') private videoRef?: ElementRef<HTMLVideoElement>;
  @ViewChild('canvasRef') private canvasRef!: ElementRef<HTMLCanvasElement>;

  private zone = inject(NgZone);
  private data: SkinData | null = null;
  private conn: Record<string, number[][]> | null = null;
  private raf = 0;

  ngAfterViewInit(): void {
    faceConn ??= fetch('skin/face_connections.json').then(r => r.json());
    faceConn.then(c => (this.conn = c)).catch(() => {});
    this.load();
    this.zone.runOutsideAngular(() => {
      const loop = () => { this.raf = requestAnimationFrame(loop); this.draw(); };
      loop();
    });
  }

  ngOnChanges(ch: SimpleChanges): void {
    if (ch['word'] || ch['skin']) this.load();
  }

  ngOnDestroy(): void { cancelAnimationFrame(this.raf); }

  private async load(): Promise<void> {
    if (this.skin) { this.data = this.skin; return; }
    this.data = null;
    if (!this.word) return;
    try {
      const r = await fetch(`${environment.apiBaseUrl}/signs/${encodeURIComponent(this.word)}/skin`);
      this.data = r.ok ? await r.json() : null;
    } catch { this.data = null; }
  }

  private draw(): void {
    const c = this.canvasRef?.nativeElement;
    if (!c) return;
    const v = this.videoRef?.nativeElement;
    if (v && v.playbackRate !== this.speed) v.playbackRate = this.speed;
    const dpr = window.devicePixelRatio || 1, W = c.clientWidth, H = c.clientHeight;
    if (c.width !== Math.round(W * dpr)) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
    const ctx = c.getContext('2d')!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const d = this.data;
    if (!this.showBones || !d || !v || v.readyState < 1) return;
    const f = d.frames[Math.min(d.n_frames - 1, Math.max(0, Math.round(v.currentTime * d.fps)))];
    if (!f) return;
    // video drawn with object-contain: map data pixels into the letterboxed picture.
    // A mirrored video's skin data is already mirrored, so no further flip here.
    const vw = v.videoWidth || d.width, vh = v.videoHeight || d.height;
    const s = Math.min(W / vw, H / vh), ox = (W - vw * s) / 2, oy = (H - vh * s) / 2;
    const sx = vw / d.width, sy = vh / d.height;
    const m = (p: number[]) => [ox + p[0] * sx * s, oy + p[1] * sy * s];
    ctx.lineCap = 'round'; ctx.lineJoin = 'round';
    if (f.p) {
      ctx.strokeStyle = 'rgba(122,108,208,0.9)'; ctx.lineWidth = 3;
      for (const [a, b] of ARM_EDGES) { const A = m(f.p[a]), B = m(f.p[b]); ctx.beginPath(); ctx.moveTo(A[0], A[1]); ctx.lineTo(B[0], B[1]); ctx.stroke(); }
    }
    if (f.f && this.conn) {
      ctx.strokeStyle = 'rgba(255,255,255,0.75)'; ctx.lineWidth = 1.2;
      ctx.beginPath();
      for (const part of FACE_PARTS) for (const [a, b] of this.conn[part] ?? []) {
        const A = m(f.f[a]), B = m(f.f[b]); ctx.moveTo(A[0], A[1]); ctx.lineTo(B[0], B[1]);
      }
      ctx.stroke();
    }
    for (const h of f.h) {
      if (!h) continue;
      ctx.lineWidth = 2.5;
      for (const [a, b] of HAND_EDGES) {
        const A = m(h[a]), B = m(h[b]);
        ctx.strokeStyle = FINGER_COLORS[FINGER(a, b)];
        ctx.beginPath(); ctx.moveTo(A[0], A[1]); ctx.lineTo(B[0], B[1]); ctx.stroke();
      }
      ctx.fillStyle = '#fff';
      for (const p of h) { const P = m(p); ctx.beginPath(); ctx.arc(P[0], P[1], 2, 0, 7); ctx.fill(); }
    }
  }
}
